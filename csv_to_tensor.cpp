
// g++ -I ~/philib/ csv_to_tensor.cpp -ldeflate -ggdb -o csv_to_tensor

#include <libdeflate.h>

#include "global.hpp"
#include "format.cpp"
#include "os_linux.cpp"
#include "hashmap.cpp"
#include "stringstore.cpp"
#include "number.cpp"
#include "csv.cpp"
#include "json_writer.cpp"

struct Task {
    Array<u8> source;
    Array<u8> target_picks;
    Array<u8> target_packs;
    Array<u8> target_rates;
    s64 out_set_packsize = -1;
    s64 out_drafts = -1;
    s64 out_seq_len = -1;
    s64 out_packs_len = -1;
};

void _csv_skip_row(Csv* csv) {
    while (csv->index < csv->data.size and csv->data[csv->index] != '\n') ++csv->index;
    ++csv->index;
}

u8 process(Stringstore* tokens, Task* task, Array_dyn<u8>* buf) {
    constexpr static s64 max_size = 15ull * 1024 * 1024 * 1024;
    constexpr static s64 seq_header = 8;
    constexpr static s64 seq_packs = 3;

    auto data_base = os_read_file(task->source);
    auto data = data_base;
    defer { array_free(&data_base); };
    os_error_panic();
    array_reserve(buf, max_size);
    
    auto* dec = libdeflate_alloc_decompressor();
    defer { libdeflate_free_decompressor(dec); };

    auto prepare_data = [&]() -> u8 {
        size_t bytes_read;
        size_t bytes_written;
        auto code = libdeflate_gzip_decompress_ex(dec, data.data, data.size, buf->data+buf->size, buf->capacity-buf->size, &bytes_read, &bytes_written);
        if (code) {
            format_print("Error: could not decompress file %a, received code %d\n", task->source, code);
            return 1;
        }
        buf->size += bytes_written;
        data = array_subarray(data, (s64)bytes_read);

        return 0;
    };

    buf->size = 0;
    if (u8 code = prepare_data()) return code;  

    Csv csv;
    csv_init(&csv, *buf);

    Array<u8> fields[] = {
        "expansion"_arr, "event_type"_arr, "draft_id"_arr, "draft_time"_arr, "rank"_arr, "event_match_wins"_arr, "event_match_losses"_arr,
        "pack_number"_arr, "pick_number"_arr, "pick"_arr, "pick_maindeck_rate"_arr, "pick_sideboard_in_rate"_arr
    };

    for (s64 i = 0; i < ARRAY_SIZE(fields); ++i) {
        auto col = csv_str(&csv);
        if (not array_equal(col, fields[i])) {
            format_print("Error: CSV file %a does not follow the expected structure. Expected field %d to be %a, but is %a\n", task->source, i, fields[i], col);
            return 2;
        }
    }

    Array_dyn<u16> card_cols;
    defer { array_free(&card_cols); };

    while (true) {
        auto col = csv_str(&csv);
        if (col.size == 0) break;

        // In older sets, pack_card_* and pool_* are interleaved (the latter are the cards that you have already picked)
        if (array_startswith(col, "pack_card_"_arr)) {
            auto card = array_subarray(col, 10);
            u16 card_id = stringstore_get_id(tokens, card);
            array_push(&card_cols, card_id);
        } else if (array_startswith(col, "pool_"_arr)) {
            array_push(&card_cols, -1);
        } else {
            break;
        }
    }

    _csv_skip_row(&csv);

    int fd_picks = os_open(task->target_picks, Os_codes::OPEN_WRITE | Os_codes::OPEN_TRUNCATE | Os_codes::OPEN_CREATE, 0664);
    int fd_packs = os_open(task->target_packs, Os_codes::OPEN_WRITE | Os_codes::OPEN_TRUNCATE | Os_codes::OPEN_CREATE, 0664);
    int fd_rates = os_open(task->target_rates, Os_codes::OPEN_WRITE | Os_codes::OPEN_TRUNCATE | Os_codes::OPEN_CREATE, 0664);

    Array<u8> cur_id = 0;
    Array_dyn<u16> cur_picks;
    Array_dyn<u16> cur_packs;
    Array_dyn<u8> cur_rates;
    defer { array_free(&cur_picks); };
    defer { array_free(&cur_packs); };
    defer { array_free(&cur_rates); };

    bool cur_skip = true;
    bool cur_written = true;
    
    s64 set_packsize = 13;
    bool first = true;

    auto tok = [tokens, &cur_picks](Array<u8> s) {
        u16 id = stringstore_get_id(tokens, s);
        array_push(&cur_picks, id);
    };
    
    Array_dyn<u8> tok_buf;
    defer { array_free(&tok_buf); };
    array_push(&tok_buf, '$');
    auto mtok = [tokens, &cur_picks, &tok_buf, &tok](Array<u8> s, char prefix = 0) {
        tok_buf.size = 1;
        if (prefix) array_push(&tok_buf, prefix);
        array_append(&tok_buf, s);
        tok(tok_buf);
    };

    s64 count_complete = 0;
    s64 count_incomplete = 0;
    s64 count_error = 0;
    s64 count_rows = 0;

    Array_dyn<u8> fd_picks_buf;
    Array_dyn<u8> fd_packs_buf;
    Array_dyn<u8> fd_rates_buf;
    defer { array_free(&fd_picks_buf); };
    defer { array_free(&fd_packs_buf); };
    defer { array_free(&fd_rates_buf); };
    
    u64 last_print = os_now();
    s64 last_print_rows = 0;

    format_print("Processing %a...\n", task->source);
    
    while (true) {
        if (csv.index+4096 >= csv.data.size and data.size) {
            array_pop_front(buf, csv.index);            
            if (u8 code = prepare_data()) return code;
            
            csv.data = *buf;
            csv.index = 0;
        }

        if (csv_empty(&csv)) break;

        if (count_rows % 17 == 0) {
            u64 now = os_now();
            if (now >= last_print + 5000000000ull) {
                s64 speed = (count_rows - last_print_rows) * 1000000000ull / (now - last_print);
                format_print("  row: %d, speed %d rows/s\n", count_rows, speed);
                last_print = now;
                last_print_rows = count_rows;
            }
        }
        
        ++count_rows;
        
        auto expansion = csv_str(&csv);
        auto event_type = csv_str(&csv);
        auto draft_id = csv_str(&csv);
        auto draft_time = csv_str(&csv);
        auto rank = csv_str(&csv);
        auto event_match_wins = csv_str(&csv);
        auto event_match_losses = csv_str(&csv);
        s64 pack_number = csv_int(&csv);
        s64 pick_number = csv_int(&csv);
        auto pick = csv_str(&csv);
        float pick_maindeck_rate = csv_float(&csv);
        float pick_sideboard_rate = csv_float(&csv);

        s64 read_games_and_rate = -1;

        if (not array_equal(cur_id, draft_id)) {
            if (not rank.size) rank = "unranked"_arr;

            if (draft_time.size < 7) {
                format_print("Error: invalid draft time '%a' (at index %d, row %d)\n", draft_time, csv.index, count_rows);
                return 8;
            }
            draft_time.size = 7;
            
            cur_id = draft_id;
            
            cur_skip = false;
            cur_written = false;
            cur_picks.size = 0;
            cur_packs.size = 0;
            cur_rates.size = 0;

            mtok(expansion);
            mtok(event_type);
            mtok(draft_time);
            mtok(rank);

            read_games_and_rate = cur_picks.size;
            array_push(&cur_picks, -1);
            array_push(&cur_picks, -1);
            
            mtok(event_match_wins, '+');
            mtok(event_match_losses, '-');

            os_error_panic();
        }

        if (cur_skip) {
            _csv_skip_row(&csv);
            continue;
        }
        
        if (cur_written) {
            format_print("Error: overlong draft (at index %d, row %d)\n", csv.index, count_rows);
            return 6;
        }
        
        s64 cur_packsize = 0;
        for (u16 card: card_cols) {
            auto col = csv_str(&csv);
            if (card == (u16)-1) continue;
            if (col.size != 1 or (u8)(col[0] - '0') > 9) {
                format_print("Error: could not parse digit column with value %a (at index %d, row %d)\n", col, csv.index, count_rows);
                return 4;
            }
            u8 count = col[0] - '0';
            cur_packsize += count;
            for (s64 i = 0; i < count; ++i) {
                array_push(&cur_packs, card);    
            }
        }

        if (read_games_and_rate != -1) {
            auto games = csv_str(&csv);
            auto win_rate = csv_str(&csv);
            if (win_rate.size > 4) win_rate.size = 4;
            
            mtok(games, 'g');
            cur_picks[read_games_and_rate] = cur_picks.back();
            --cur_picks.size;
            
            mtok(win_rate, 'w');
            cur_picks[read_games_and_rate+1] = cur_picks.back();
            --cur_picks.size;
        }
        
        _csv_skip_row(&csv);

        if (set_packsize < cur_packsize) {
            if (not first) {
                format_print("Error: set_packsize is not stable! was %d, now set to %d (at index %d)\n", set_packsize, cur_packsize, csv.index);
                return 3;
            }
            set_packsize = cur_packsize;
        }
        
        s64 pick_it = pack_number * set_packsize + pick_number;
        s64 it = seq_header + pick_it;
        if (cur_picks.size != it) {
            cur_skip = true;
            ++count_incomplete;
            continue;
        }

        tok(pick);
        array_push(&cur_rates, (u8)(pick_maindeck_rate * 255.f));

        if (cur_packsize != set_packsize - pick_number) {
            cur_skip = true;
            ++count_error;
            //if (count_error < 100) {
            //    format_print("Error: bad number of cards in pack, expected %d - %d = %d, got %d (at index %d, row %d)\n", set_packsize, pick_number, set_packsize - pick_number, cur_packsize, csv.index, count_rows);
            //}
            //return 5;
            continue;
        }

        if (pick_it == set_packsize * seq_packs-1) {
            array_append(&fd_picks_buf, array_bytes_arr(cur_picks));
            array_append(&fd_packs_buf, array_bytes_arr(cur_packs));
            array_append(&fd_rates_buf, array_bytes_arr(cur_rates));
            if (fd_picks_buf.size >= 4096 * 16) {
                os_write_fixed(fd_picks, fd_picks_buf);
                fd_picks_buf.size = 0;
            }
            if (fd_packs_buf.size >= 4096 * 16) {
                os_write_fixed(fd_packs, fd_packs_buf);
                fd_packs_buf.size = 0;
            }
            if (fd_rates_buf.size >= 4096 * 16) {
                os_write_fixed(fd_rates, fd_rates_buf);
                fd_rates_buf.size = 0;
            }

            ++count_complete;
            cur_written = true;
        }
    }
    
    os_write_fixed(fd_picks, fd_picks_buf);
    os_write_fixed(fd_packs, fd_packs_buf);
    os_write_fixed(fd_rates, fd_rates_buf);
    
    os_close(fd_picks);
    os_close(fd_packs);
    os_close(fd_rates);

    task->out_set_packsize = set_packsize;
    task->out_seq_len = seq_header + seq_packs * set_packsize;
    task->out_drafts = count_complete;
    task->out_packs_len = seq_packs * set_packsize * (set_packsize+1) / 2;
    
    
    format_print("Done. (rows: %d, complete: %d, incomplete %d, error %d)\n", count_rows, count_complete, count_incomplete, count_error);
    
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        format_print("Usage: ./csv_to_tensor <dir>\n");
        return 1;
    }
    
    os_init();
    
    Array<u8> dir = array_from_str(argv[1]);
    auto suffix = ".csv.gz"_arr;

    Dir_entries entries;
    if (array_endswith(dir, suffix)) {
        s64 index = 0;
        for (s64 i = dir.size-1; i >= 0; --i) {
            if (dir[i] == '/') {
                index = i;
                break;
            }
        }

        auto name = array_subarray(dir, index+1);
        dir = array_subarray(dir, 0, index);
        if (dir.size == 0) dir = "."_arr;
        
        array_push(&entries.entries, {array_append(&entries.name_data, name), Dir_entries::FILE});
        
    } else {
        os_directory_list(dir, &entries);
    }

    Array_dyn<Task> tasks;
    Array_dyn<u8> buf;

    
    for (auto i: entries.entries) {
        if (i.type != Dir_entries::FILE) continue;
        auto name = array_suboffset(entries.name_data, i.name);

        if (not array_endswith(name, suffix)) continue;

        Task task;
        buf.size = 0;
        format_into(&buf, "%a/%a", dir, name);
        task.source = array_copy(buf);

        auto name_stem = array_subarray(name, 0, name.size - suffix.size);
        
        buf.size = 0;
        format_into(&buf, "tensors/%a_picks.dat", name_stem);
        task.target_picks = array_copy(buf);
        
        buf.size = 0;
        format_into(&buf, "tensors/%a_packs.dat", name_stem);
        task.target_packs = array_copy(buf);
        
        buf.size = 0;
        format_into(&buf, "tensors/%a_rates.dat", name_stem);
        task.target_rates = array_copy(buf);
        

        array_push(&tasks, task);
    }

    Stringstore tokens;
    stringstore_get_id(&tokens, ""_arr); // Make sure that the empty token has id 0
    
    for (auto& task: tasks) {
        u8 code = process(&tokens, &task, &buf);
        if (code) return code;
    }

    Json_writer writer;
    json_writer_init(&writer);
    defer { json_writer_free(&writer); };

    {
        auto _ = json_writer_obj(&writer);
        json_writer_attr_key(&writer, "tokens");
        {
            auto _tokens = json_writer_obj(&writer);
            for (s64 i = 0; i < tokens.size; ++i) {
                json_writer_attr(&writer, stringstore_get_str(&tokens, i), i);
            }
        }
        json_writer_attr_key(&writer, "data");
        {
            auto _data = json_writer_arr(&writer);
            for (auto task: tasks) {
                auto _task = json_writer_obj(&writer);
                json_writer_attr(&writer, "packs", task.target_packs);
                json_writer_attr(&writer, "picks", task.target_picks);
                json_writer_attr(&writer, "rates", task.target_rates);
                json_writer_attr(&writer, "set_packsize", task.out_set_packsize);
                json_writer_attr(&writer, "seq_len", task.out_seq_len);
                json_writer_attr(&writer, "packs_len", task.out_packs_len);
                json_writer_attr(&writer, "n_drafts", task.out_drafts);
            }
        }
    }

    array_write_to_file("datasets.json"_arr, writer.output);
    
}
