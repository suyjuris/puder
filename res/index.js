const pageData = JSON.parse(document.getElementById('pageData').textContent);


// API call helper function
function apiCall(endpoint, params) {
    for (var i of document.querySelectorAll("button")) {
        i.disabled = true;
    }
    
    fetch('.' + endpoint, {
        method: 'POST',
        body: JSON.stringify(params),
    })
    .then(response => {
        response.json().then(data => {
            if (data.result == 'success') {
                window.location.href = data.location;
            } else {
                alert('Error: ' + data.message);
            }
        });
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during the API call.');
    })
   .finally(() => {
        for (var i of document.querySelectorAll("button")) {
            i.disabled = false;
        }
   });
}

// Function to populate expansion dropdown
function populateExpansions() {
    const select = document.getElementById('expansion-select');
    
    // Clear existing options except the first one
    while (select.children.length > 1) {
        select.removeChild(select.lastChild);
    }
    
    // Add expansion options
    pageData.expansions.forEach(expansion => {
        const option = document.createElement('option');
        option.value = expansion;
        option.textContent = expansion;
        select.appendChild(option);
    });
}

// Function to format date for time input (YYYY-MM-DD format)
function formatDateForInput(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

// Function to handle create draft form submission
function handleCreateDraft(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    
    const expansion = formData.get('expansion');
    if (!expansion) {
        alert('Please choose an expansion.');
        return;
    }
    
    const params = {
        headers: {expansion: expansion}
    };
    
    apiCall('/create', params);
}

// Function to handle load random draft
function handleLoadRandom() {
    event.preventDefault();
    apiCall('/load_random', {'dummy': 0});
}

function handleImportDraft(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    
    const link = formData.get('link');
    if (!link) {
        alert('Please provide the link.');
        return;
    }
    
    const params = {
        link: link
    };
    
    apiCall('/import_draft', params);
}


// Initialize page when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Populate expansions dropdown
    populateExpansions();
    
    // Add event listeners
    document.getElementById('create-draft-form').addEventListener('submit', handleCreateDraft);
    document.getElementById('load-random-form').addEventListener('submit', handleLoadRandom);
    document.getElementById('import-draft-form').addEventListener('submit', handleImportDraft);
});
