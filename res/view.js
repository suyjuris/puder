
const pageData = JSON.parse(document.getElementById('pageData').textContent);

// Helper function to update URL parameters
function updateURL(pack, pick = null) {
    const url = new URL(window.location);
    url.searchParams.set('pack', pack);
    if (pick !== null) {
        url.searchParams.set('pick', pick);
    } else {
        url.searchParams.delete('pick');
    }
    window.history.replaceState({}, '', url);
}

// Handle pack tab switching
function showPack(packId) {
    // Hide all pack contents
    const packContents = document.querySelectorAll('.pack-content');
    packContents.forEach(content => {
        content.classList.remove('active');
    });
    
    // Deactivate all pack tabs
    const packTabs = document.querySelectorAll('.pack-tab');
    packTabs.forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show the selected pack content and activate tab
    document.getElementById('pack-content-' + packId).classList.add('active');
    document.getElementById('pack-tab-' + packId).classList.add('active');
    
    // Update URL with pack parameter
    updateURL(packId);
    
    // Find the first pick in this pack and show it (but only for non-deckbuilding packs)
    if (packId != 3) {
        const firstPickTab = document.querySelector(`#pack-content-${packId} .pick-tab`);
        if (firstPickTab) {
            firstPickTab.click();
        }
    }
}

// Handle pick tab switching
function showPick(pickId, packId) {
    // Hide all pick contents within the current pack
    const currentPack = document.getElementById('pack-content-' + packId);
    const pickContents = currentPack.querySelectorAll('.pick-content');
    pickContents.forEach(content => {
        content.classList.remove('active');
    });
    
    // Deactivate all pick tabs
    const pickTabs = currentPack.querySelectorAll('.pick-tab');
    pickTabs.forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show the selected pick content and activate tab
    document.getElementById('pick-content-' + pickId).classList.add('active');
    document.getElementById('pick-tab-' + pickId).classList.add('active');
    
    // Update URL with pack and pick parameters
    const pickWithinPack = pickId % pageData.n_packsize;
    updateURL(packId, pickWithinPack);
}

// Function to remove a card from the pack
function removeCard(event, round, pick, card) {
    // Stop event propagation to prevent card preview
    event.stopPropagation();

    const cardIndex = pageData.packs[round][pick].indexOf(card);
    const cards = [{'index': cardIndex, 'card': ''}];
    apiCall('/pack', {'rnd': round, 'pick': pick, 'cards': cards});
}

function pickCard(round, pick, card) {
    apiCall('/pick', {'rnd': round, 'pick': pick, 'pick_card': card});
}

function apiCall(endpoint, params) {
    // Submit the request to the server
    params.id = pageData.view_id;
    fetch('.' + endpoint, {
        method: 'POST',
        body: JSON.stringify(params),
    })
    .then(response => {
        response.json().then(data => {
            if (data.result == 'success') {
                if (data.location == 'reload') {
                    window.location.reload();
                } else {
                    window.location.href = data.location;
                }
            } else {
                alert('Error: ' + data.message);
            }
        });
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during the API call.');
    });
}

// Function to update a header value
function updateHeader(selectElement, index) {
    const tokenId = parseInt(selectElement.value, 10);
    apiCall('/header', { 'index': index, 'token': tokenId });
}

// Function to store current weights
function storeWeights() {
    apiCall('/store_weights', {});
}

// Function to clear stored weights
function clearWeights() {
    apiCall('/store_weights', {'clear': true});
}


// Function to show card preview on hover
function showCardPreview(event, src) {
    const preview = document.getElementById('card-preview');
    const previewImg = document.getElementById('preview-img');
    
    // Set the image source
    previewImg.src = src;
    
    // Calculate position (to left of cursor)
    const xPos = event.clientX - 360; // Card width (260px) + some margin
    const yPos = event.clientY - 100; // Position a bit above the cursor
    
    // Update position and show preview
    preview.style.left = `${xPos}px`;
    preview.style.top = `${yPos}px`;
    preview.style.display = 'block';
    
    // Preload the image
    previewImg.onload = function() {
        // Adjust position if the preview would go off the screen
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const previewWidth = preview.offsetWidth;
        const previewHeight = preview.offsetHeight;
        
        // Check if preview goes off left edge
        if (xPos < 0) {
            preview.style.left = `${event.clientX + 20}px`; // Move to right of cursor instead
        }
        
        // Check if preview goes off bottom edge
        if (yPos + previewHeight > viewportHeight) {
            preview.style.top = `${viewportHeight - previewHeight - 20}px`;
        }
        
        // Check if preview goes off top edge
        if (yPos < 0) {
            preview.style.top = '20px';
        }
    };
}

// Function to hide card preview
function hideCardPreview() {
    const preview = document.getElementById('card-preview');
    preview.style.display = 'none';
}

// Function to open the card add modal
function openCardAddModal(round, pickIdx) {
    const modal = document.getElementById('card-add-modal');
    const form = document.getElementById('card-add-form');
    
    // Set data attributes for the form submission
    form.setAttribute('data-round', round);
    form.setAttribute('data-pick', pickIdx);
    
    // Clear any previous input
    document.getElementById('card-input-area').value = '';
    
    // Show the modal
    modal.classList.add('active');
}

// Function to close the card add modal
function closeCardAddModal() {
    const modal = document.getElementById('card-add-modal');
    modal.classList.remove('active');
}

// Function to submit the card add form
function submitCardAddForm() {
    const form = document.getElementById('card-add-form');
    const textarea = document.getElementById('card-input-area');
    const round = form.getAttribute('data-round');
    const pick = form.getAttribute('data-pick');

    var currentCardCount = 0;
    for (const i of pageData.packs[round][pick]) {
        currentCardCount += i != 0;
    }
    
    // Parse the card input (one card per line)
    const cardLines = textarea.value.trim().split('\n');

    // Add card entries to the form data, starting from the current count
    const cards = cardLines.map((line, index) => ({'index': currentCardCount + index, 'card': line.trim()}));
    apiCall('/pack', {'rnd': round, 'pick': pick, 'cards': cards});
}

// Function to format select display text (removing weights)
function formatSelectDisplayText() {
    const metaDivs = document.querySelectorAll('.meta-value');
    
    metaDivs.forEach(div => {
        const select = div.querySelector('select');
        const span = div.querySelector('span');

        select.style.width = (span.clientWidth + 40) + 'px';
    });
}

// Initialize - show the first pack when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners for the modal
    document.getElementById('close-modal-btn').addEventListener('click', closeCardAddModal);
    document.getElementById('submit-cards-btn').addEventListener('click', submitCardAddForm);
    
    // Close modal when clicking outside of it
    document.getElementById('card-add-modal').addEventListener('click', function(event) {
        if (event.target === this) {
            closeCardAddModal();
        }
    });

    let full_prefix = true;

    document.getElementById('card-input-area').addEventListener('input', (e) => {
        if (e.inputType.startsWith('delete')) return;
        const textarea = e.target;
        const text = textarea.value;
        const pos = textarea.selectionStart;
        let i = pos;
        while (i > 0 && text[i-1] != '\n') i--;
        const word = text.substring(i, pos);
        if (word.length == 0) return;

        let has_prefix = false;
        full_prefix = true;
        let prefix = undefined;
        for (const w of pageData['cards']) {
            if (!w.startsWith(word)) continue;
            if (has_prefix) {
                while (!w.startsWith(prefix)) {
                    prefix = prefix.substring(0, prefix.length-1);
                    full_prefix = false;
                }
            } else {
                prefix = w;
                has_prefix = true;
            }
        }
        if (!has_prefix) return;
        if (prefix.length <= word.length+1) return;

        textarea.value += prefix.substring(word.length)
        textarea.selectionStart = pos;
        textarea.selectionEnd = textarea.value.length;
    });
    document.getElementById('card-input-area').addEventListener('beforeinput', (e) => {
        const textarea = e.target;
        if (e.inputType == 'insertLineBreak' && textarea.selectionStart < textarea.selectionEnd) {
            if (full_prefix) {
                textarea.value += '\n';
            } else {
                textarea.selectionStart = textarea.selectionEnd;
            }
            e.preventDefault()
        }
        full_prefix = true;
    });

    
    // Format the select dropdowns
    formatSelectDisplayText();
});
