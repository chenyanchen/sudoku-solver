// Sudoku Solver Frontend Application

const KEEP_IMAGE_KEY = 'sudoku-keep-image';
const I18N = window.I18N || {};

// Localized string with {name} placeholders filled from `values`.
function t(key, values = {}) {
    return (I18N[key] || key).replace(/\{(\w+)\}/g, (_, name) => values[name] ?? '');
}

function storageGet(key) {
    try { return localStorage.getItem(key); } catch { return null; }
}

function storageSet(key, value) {
    try { localStorage.setItem(key, value); } catch { /* private mode */ }
}

class SudokuSolver {
    constructor() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.keepImage = document.getElementById('keepImage');
        this.previewSection = document.getElementById('previewSection');
        this.originalImage = document.getElementById('originalImage');
        this.detectedCard = document.getElementById('detectedCard');
        this.detectedImage = document.getElementById('detectedImage');
        this.solveButton = document.getElementById('solveButton');
        this.resolveButton = document.getElementById('resolveButton');
        this.resultsSection = document.getElementById('resultsSection');
        this.messageSection = document.getElementById('messageSection');
        this.stats = document.getElementById('stats');
        this.selectedFile = null;
        this.token = null;           // authorizes feedback on the current reading
        this.recognized = null;      // givens as recognized

        this.init();
    }

    init() {
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        this.keepImage.checked = storageGet(KEEP_IMAGE_KEY) === '1';
        this.keepImage.addEventListener('change', () => {
            storageSet(KEEP_IMAGE_KEY, this.keepImage.checked ? '1' : '0');
        });

        this.solveButton.addEventListener('click', () => this.solvePuzzle());
        this.resolveButton.addEventListener('click', () => this.solveCorrected());
        this.loadStats();
    }

    async loadStats() {
        try {
            const data = await (await fetch('/api/v1/stats')).json();
            if (data.total > 0) {
                this.stats.textContent = t('stats', data);
                this.stats.hidden = false;
            }
        } catch {
            // stats are informational only
        }
    }

    handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            this.showMessage(t('notImage'), 'error');
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            this.showMessage(t('tooLarge'), 'error');
            return;
        }

        this.selectedFile = file;
        this.clearMessages();

        const reader = new FileReader();
        reader.onload = (e) => {
            this.originalImage.src = e.target.result;
            this.previewSection.style.display = 'block';
            this.resultsSection.style.display = 'none';
            this.detectedCard.style.display = 'none';
            this.solvePuzzle();
        };
        reader.readAsDataURL(file);
    }

    async solvePuzzle() {
        if (!this.selectedFile) {
            this.showMessage(t('noImage'), 'error');
            return;
        }

        this.setLoading(true);
        this.clearMessages();
        this.token = null;

        try {
            const formData = new FormData();
            formData.append('image', this.selectedFile);
            formData.append('keep_image', this.keepImage.checked ? 'true' : 'false');

            const recResp = await fetch('/api/v1/sudoku:recognize', { method: 'POST', body: formData });
            if (!recResp.ok) {
                const err = await recResp.json().catch(() => ({}));
                this.showMessage(err.detail || t('uploadFailed', { status: recResp.status }), 'error');
                return;
            }
            const recData = await recResp.json();
            this.token = recData.token;

            if (recData.detected_image) {
                this.displayDetectedGrid(recData.detected_image);
            }
            if (!recData.success || !recData.grid) {
                this.showMessage(t('noGrid'), 'warning');
                return;
            }

            this.recognized = recData.grid;
            this.displayEditableGrid(recData.grid);
            this.displayGrid(null, 'solvedGrid');
            this.resultsSection.style.display = 'block';

            if (recData.solved) {
                await this.solve(recData.grid, t('solved'));
            } else {
                this.showMessage(t('misread'), 'warning');
            }
        } catch (error) {
            this.showMessage(t('network', { error: error.message }), 'error');
        } finally {
            this.setLoading(false);
            this.loadStats();
        }
    }

    async solveCorrected() {
        const grid = this.readEditableGrid();
        if (!grid) {
            this.showMessage(t('digitsOnly'), 'error');
            return;
        }
        if (this.token) {
            // The corrected givens are training data; failures here do not block solving.
            fetch('/api/v1/sudoku:feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token: this.token, grid: { cells: grid } }),
            }).then(() => this.loadStats()).catch(() => {});
        }
        await this.solve(grid, t('solvedCorrected'));
    }

    async solve(grid, successMessage) {
        const resp = await fetch('/api/v1/sudoku:solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grid: { cells: grid } }),
        });
        const data = await resp.json();
        if (data.success && data.solved) {
            this.showMessage(successMessage, 'success');
            this.displayGrid(data.solved, 'solvedGrid', grid);
        } else {
            const reason = { no_solution: 'noSolution', multiple_solutions: 'multipleSolutions', invalid: 'invalid' };
            this.showMessage(t(reason[data.status] || 'noSolution'), 'warning');
            this.displayGrid(null, 'solvedGrid');
        }
    }

    displayDetectedGrid(base64Image) {
        this.detectedImage.src = 'data:image/png;base64,' + base64Image;
        this.detectedCard.style.display = 'block';
    }

    displayEditableGrid(grid) {
        const container = document.getElementById('originalGrid');
        container.innerHTML = '';
        for (let row = 0; row < 9; row++) {
            for (let col = 0; col < 9; col++) {
                const input = document.createElement('input');
                input.className = 'sudoku-cell given';
                input.inputMode = 'numeric';
                input.maxLength = 1;
                input.setAttribute('aria-label', t('cell', { row: row + 1, col: col + 1 }));
                input.value = grid[row][col] || '';
                input.addEventListener('focus', () => input.select());
                input.addEventListener('input', () => {
                    input.value = input.value.replace(/[^1-9]/g, '');
                    input.classList.toggle('edited', (Number(input.value) || 0) !== grid[row][col]);
                    this.resolveButton.disabled = false;
                });
                container.appendChild(input);
            }
        }
        this.resolveButton.disabled = true;
    }

    readEditableGrid() {
        const inputs = document.querySelectorAll('#originalGrid input');
        const grid = [];
        for (let i = 0; i < 81; i++) {
            const value = inputs[i].value === '' ? 0 : Number(inputs[i].value);
            if (!Number.isInteger(value) || value < 0 || value > 9) {
                return null;
            }
            if (i % 9 === 0) grid.push([]);
            grid[grid.length - 1].push(value);
        }
        return grid;
    }

    displayGrid(gridData, elementId, givens = null) {
        const container = document.getElementById(elementId);
        container.innerHTML = '';
        for (let row = 0; row < 9; row++) {
            for (let col = 0; col < 9; col++) {
                const cell = document.createElement('div');
                cell.className = 'sudoku-cell';
                const value = gridData ? gridData[row][col] : 0;
                if (value !== 0) {
                    cell.textContent = value;
                    cell.classList.add(givens && givens[row][col] ? 'given' : 'solved');
                }
                container.appendChild(cell);
            }
        }
    }

    setLoading(loading) {
        const btnText = this.solveButton.querySelector('.btn-text');
        const btnLoader = this.solveButton.querySelector('.btn-loader');
        this.solveButton.disabled = loading;
        btnText.style.display = loading ? 'none' : 'inline';
        btnLoader.style.display = loading ? 'inline-flex' : 'none';
    }

    showMessage(message, type = 'info') {
        this.clearMessages();
        const messageEl = document.createElement('div');
        messageEl.className = `message message-${type}`;
        messageEl.textContent = message;
        this.messageSection.appendChild(messageEl);
    }

    clearMessages() {
        this.messageSection.innerHTML = '';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new SudokuSolver();
});
