let parkingData = [];
let csvHeaders = [];
let sortState = { column: null, ascending: true };
let searchTerm = '';
let filteredData = [];
let searchTimeout;
let selectedViolations = {};

document.addEventListener('DOMContentLoaded', function() {
    loadCSV();
});

function loadCSV() {
    fetch('data/violations_by_street.csv')
        .then(response => response.text())
        .then(data => {
            const parsed = parseCSV(data);
            csvHeaders = parsed.headers;
            parkingData = parsed.data;
            filteredData = [...parkingData];
            generateTableHeaders();
            renderTable();
            setupEventListeners();
        })
        .catch(error => console.error('Error loading CSV:', error));
}

function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const allHeaders = lines[0].split(',').map(h => h.trim());
    
    // Remove street_name from headers (it's displayed as the sticky first column)
    const headers = allHeaders.slice(1);
    
    const data = [];
    
    // Process each row starting from line 1 (skip header)
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        // Simple CSV parsing - handles basic cases
        const values = parseCSVLine(line);
        if (values.length > 0) {
            const streetName = values[0];
            
            // Skip rows with invalid street names
            if (!streetName || streetName === '' || streetName.length < 1) continue;
            
            const row = {
                street: streetName,
                violations: {}
            };
            
            // Map each violation type to its count (skip first value which is street name)
            for (let j = 1; j < values.length && j < allHeaders.length; j++) {
                const count = parseInt(values[j]) || 0;
                row.violations[allHeaders[j]] = count;
            }
            
            data.push(row);
        }
    }
    
    return { headers, data };
}

function parseCSVLine(line) {
    const values = [];
    let current = '';
    let insideQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            insideQuotes = !insideQuotes;
        } else if (char === ',' && !insideQuotes) {
            values.push(current.trim().replace(/^"|"$/g, ''));
            current = '';
        } else {
            current += char;
        }
    }
    
    values.push(current.trim().replace(/^"|"$/g, ''));
    return values;
}

function generateTableHeaders() {
    const thead = document.getElementById('tableHead');
    const headerRow = document.createElement('tr');
    
    // Add street name header (always first, not sortable in same way)
    const streetHeader = document.createElement('th');
    streetHeader.className = 'sortable street-name-header';
    streetHeader.setAttribute('data-column', '0');
    streetHeader.innerHTML = 'Street Name <span class="sort-icon">▼</span>';
    headerRow.appendChild(streetHeader);
    
    // Add violation type headers
    csvHeaders.forEach((header, index) => {
        const th = document.createElement('th');
        th.className = 'sortable';
        th.setAttribute('data-column', index + 1);
        th.setAttribute('data-violation', header);
        th.innerHTML = `${header} <span class="sort-icon">▼</span>`;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    document.getElementById('totalCount').textContent = parkingData.length;
    
    // Initialize violation filter
    initializeViolationFilter();
}

function renderTable() {
    const tableBody = document.getElementById('tableBody');
    
    // Build HTML string instead of creating elements one by one
    let html = '';
    filteredData.forEach(row => {
        html += '<tr class="data-row"><td class="street-cell">' + row.street + '</td>';
        
        csvHeaders.forEach(header => {
            const count = row.violations[header] || 0;
            html += '<td class="count-cell" data-violation="' + header + '">' + count + '</td>';
        });
        
        html += '</tr>';
    });
    
    tableBody.innerHTML = html;
    updateRowCount();
    updateColumnVisibility();
}

function setupEventListeners() {
    const searchInput = document.getElementById('searchInput');
    // Debounce search to prevent excessive filtering
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => handleSearch(e), 150);
    }, { passive: true });
    
    const sortHeaders = document.querySelectorAll('.sortable');
    sortHeaders.forEach(header => {
        header.addEventListener('click', handleSort);
    });
    
    // Dropdown functionality
    const dropdownBtn = document.getElementById('violationDropdownBtn');
    const dropdownMenu = document.getElementById('violationDropdown');
    
    dropdownBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdownMenu.style.display = dropdownMenu.style.display === 'none' ? 'block' : 'none';
    });
    
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.dropdown-container')) {
            dropdownMenu.style.display = 'none';
        }
    });
    
    document.getElementById('selectAllBtn').addEventListener('click', selectAllViolations);
    document.getElementById('deselectAllBtn').addEventListener('click', deselectAllViolations);
}

function handleSearch(e) {
    searchTerm = e.target.value.toLowerCase();
    
    filterAndRenderTable();
}

function handleSort(e) {
    const columnIndex = parseInt(e.currentTarget.getAttribute('data-column'));
    
    // Toggle sort direction if same column clicked
    if (sortState.column === columnIndex) {
        sortState.ascending = !sortState.ascending;
    } else {
        sortState.column = columnIndex;
        sortState.ascending = false;  // Default to descending (largest first)
    }
    
    // Sort filteredData array in-memory (not DOM)
    filteredData.sort((a, b) => {
        let aVal, bVal;
        
        if (columnIndex === 0) {
            // Street name column
            aVal = a.street;
            bVal = b.street;
        } else {
            // Violation columns
            const header = csvHeaders[columnIndex - 1];
            aVal = a.violations[header] || 0;
            bVal = b.violations[header] || 0;
        }
        
        // Numeric comparison
        if (typeof aVal === 'number' && typeof bVal === 'number') {
            return sortState.ascending ? aVal - bVal : bVal - aVal;
        }
        
        // String comparison
        aVal = String(aVal).toLowerCase();
        bVal = String(bVal).toLowerCase();
        return sortState.ascending 
            ? aVal.localeCompare(bVal) 
            : bVal.localeCompare(aVal);
    });
    
    // Render sorted table
    renderTable();
    
    // Update sort indicators
    document.querySelectorAll('.sortable .sort-icon').forEach(icon => {
        icon.textContent = '▼';
        icon.style.opacity = '0.3';
    });
    
    const activeIcon = e.currentTarget.querySelector('.sort-icon');
    activeIcon.style.opacity = '1';
    activeIcon.textContent = sortState.ascending ? '▲' : '▼';
}

function updateRowCount() {
    document.getElementById('rowCount').textContent = filteredData.length;
}

function initializeViolationFilter() {
    const checkboxContainer = document.getElementById('violationCheckboxes');
    checkboxContainer.innerHTML = '';
    
    // Initialize all violations as selected by default
    csvHeaders.forEach(violation => {
        selectedViolations[violation] = true;
        
        const label = document.createElement('label');
        label.className = 'violation-checkbox-item';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.value = violation;
        checkbox.addEventListener('change', applyViolationFilter);
        
        const labelText = document.createElement('label');
        labelText.textContent = violation;
        
        label.appendChild(checkbox);
        label.appendChild(labelText);
        checkboxContainer.appendChild(label);
    });
}

function selectAllViolations() {
    csvHeaders.forEach(violation => {
        selectedViolations[violation] = true;
    });
    
    document.querySelectorAll('#violationCheckboxes input[type="checkbox"]').forEach(cb => {
        cb.checked = true;
    });
    
    applyViolationFilter();
}

function deselectAllViolations() {
    csvHeaders.forEach(violation => {
        selectedViolations[violation] = false;
    });
    
    document.querySelectorAll('#violationCheckboxes input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    
    applyViolationFilter();
}

function applyViolationFilter() {
    // Update selectedViolations from checkboxes
    document.querySelectorAll('#violationCheckboxes input[type="checkbox"]').forEach(cb => {
        selectedViolations[cb.value] = cb.checked;
    });
    
    // Filter data
    filterAndRenderTable();
}

function filterAndRenderTable() {
    filteredData = parkingData.filter(row => {
        // Check search term
        const matchesSearch = row.street.toLowerCase().includes(searchTerm);
        
        // Check if row has any selected violations with count > 0
        const hasSelectedViolation = csvHeaders.some(violation => {
            return selectedViolations[violation] && (row.violations[violation] || 0) > 0;
        });
        
        return matchesSearch && hasSelectedViolation;
    });
    
    renderTable();
    document.getElementById('noResults').style.display = filteredData.length === 0 ? 'block' : 'none';
}

function updateColumnVisibility() {
    // Hide/show headers based on selected violations
    csvHeaders.forEach(violation => {
        const headers = document.querySelectorAll(`th[data-violation="${violation}"]`);
        const cells = document.querySelectorAll(`td[data-violation="${violation}"]`);
        
        const isHidden = !selectedViolations[violation];
        
        headers.forEach(header => {
            header.style.display = isHidden ? 'none' : '';
        });
        
        cells.forEach(cell => {
            cell.style.display = isHidden ? 'none' : '';
        });
    });
}
