// Sudoku Solver Frontend Application

const I18N = window.I18N || {};
// Where the page and the API live; both empty on our server (see i18n.render).
const SITE = window.SITE || { prefix: '', api: '' };

// Localized string with {name} placeholders filled from `values`.
function t(key, values = {}) {
    return (I18N[key] || key).replace(/\{(\w+)\}/g, (_, name) => values[name] ?? '');
}

// Grids are 9x9 or 16x16; a 16x16 grid's values 10-16 are shown as A-G.
const SYMBOLS = '123456789ABCDEFG';
const symbol = (value) => (value ? SYMBOLS[value - 1] : '');
const boxSide = (size) => Math.round(Math.sqrt(size));

// Value of a typed character in a grid of `size`, or 0 if it is not one.
function parseSymbol(text, size) {
    const index = SYMBOLS.indexOf(text.toUpperCase());
    return index >= 0 && index < size ? index + 1 : 0;
}

// Cell classes that draw the thicker lines between boxes.
function boxClasses(cell, row, col, size) {
    const box = boxSide(size);
    if (col % box === box - 1 && col < size - 1) cell.classList.add('box-right');
    if (row % box === box - 1 && row < size - 1) cell.classList.add('box-bottom');
}

function layoutGrid(container, size) {
    container.innerHTML = '';
    container.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;
    container.classList.toggle('size-16', size === 16);
}

class SudokuSolver {
    constructor() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.previewSection = document.getElementById('previewSection');
        this.originalImage = document.getElementById('originalImage');
        this.detectedCard = document.getElementById('detectedCard');
        this.detectedImage = document.getElementById('detectedImage');
        this.solveButton = document.getElementById('solveButton');
        this.resolveButton = document.getElementById('resolveButton');
        this.resultsSection = document.getElementById('resultsSection');
        this.messageSection = document.getElementById('messageSection');
        this.stats = document.getElementById('stats');
        this.manualButton = document.getElementById('manualButton');
        this.shareButton = document.getElementById('shareButton');
        this.sizeButton = document.getElementById('sizeButton');
        this.puzzleTitle = document.getElementById('puzzleTitle');
        this.puzzleHint = document.getElementById('puzzleHint');
        this.selectedFile = null;
        this.token = null;           // authorizes feedback on the current reading
        this.recognized = null;      // givens as recognized
        this.manual = false;         // the puzzle was typed in, not recognized
        this.solution = null;        // { givens, solved } of the last solution shown
        this.size = 9;               // cells per side of the puzzle shown

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

        this.solveButton.addEventListener('click', () => this.solvePuzzle());
        this.resolveButton.addEventListener('click', () => this.solveCorrected());
        this.manualButton.addEventListener('click', () => this.startManual());
        this.sizeButton.addEventListener('click', () => this.startManual(this.size === 9 ? 16 : 9));
        this.shareButton.addEventListener('click', () => this.shareSolution());
        this.loadStats();

        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register(SITE.prefix + '/sw.js').catch(() => {});
        }
        if (new URLSearchParams(location.search).has('shared')) {
            history.replaceState(null, '', location.pathname);
            this.loadSharedImage();
        } else if (location.hash === '#manual') {
            this.startManual();
        }
    }

    // An image shared to the installed app, left in the cache by sw.js.
    async loadSharedImage() {
        try {
            const cache = await caches.open('shared-image');
            const response = await cache.match('/shared-image');
            if (!response) return;
            await cache.delete('/shared-image');
            const blob = await response.blob();
            const type = blob.type || 'image/png';
            this.handleFileSelect(new File([blob], 'shared.' + type.split('/')[1], { type }));
        } catch {
            // the upload area still works
        }
    }

    startManual(size = 9) {
        this.manual = true;
        this.token = null;
        this.selectedFile = null;
        this.clearMessages();
        this.previewSection.style.display = 'none';
        this.size = size;
        this.setPuzzleMode();
        this.displayEditableGrid(Array.from({ length: size }, () => Array(size).fill(0)));
        this.displayGrid(null, 'solvedGrid');
        this.resolveButton.disabled = false;
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        document.querySelector('#originalGrid input').focus({ preventScroll: true });
    }

    setPuzzleMode() {
        this.puzzleTitle.textContent = t(this.manual ? 'manualTitle' : 'recognizedTitle');
        const manualHint = this.size === 16 ? 'manualHint16' : 'manualHint';
        this.puzzleHint.textContent = t(this.manual ? manualHint : 'recognizedHint');
        this.sizeButton.hidden = !this.manual;
        this.sizeButton.textContent = t(this.size === 9 ? 'size16' : 'size9');
        this.resolveButton.textContent = t(this.manual ? 'manualSolve' : 'resolve');
        this.shareButton.hidden = true;
        this.solution = null;
    }

    async loadStats() {
        try {
            const data = await (await fetch(SITE.api + '/api/v1/stats')).json();
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
        this.manual = false;
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

            const recResp = await fetch(SITE.api + '/api/v1/sudoku:recognize', { method: 'POST', body: formData });
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
            this.setPuzzleMode();
            this.displayEditableGrid(recData.grid);
            this.displayGrid(null, 'solvedGrid');
            this.resultsSection.style.display = 'block';

            await this.solve(recData.grid, t('solved'));
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
            this.showMessage(t(this.size === 16 ? 'symbolsOnly16' : 'digitsOnly'), 'error');
            return;
        }
        if (this.manual && this.size === 9 && grid.flat().filter(Boolean).length < 17) {
            this.showMessage(t('tooFewGivens'), 'warning');
            return;
        }
        if (this.token) {
            // The corrected givens are training data; failures here do not block solving.
            fetch(SITE.api + '/api/v1/sudoku:feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token: this.token, grid: { cells: grid } }),
            }).then(() => this.loadStats()).catch(() => {});
        }
        await this.solve(grid, t(this.manual ? 'solved' : 'solvedCorrected'));
    }

    async solve(grid, successMessage) {
        const resp = await fetch(SITE.api + '/api/v1/sudoku:solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grid: { cells: grid } }),
        });
        const data = await resp.json();
        if (data.success && data.solved) {
            this.showMessage(successMessage, 'success');
            this.displayGrid(data.solved, 'solvedGrid', grid);
            this.solution = { givens: grid, solved: data.solved };
            this.shareButton.hidden = false;
        } else {
            const reason = { no_solution: 'noSolution', invalid: 'invalid' };
            this.showMessage(t(reason[data.status] || 'noSolution'), 'warning');
            this.displayGrid(null, 'solvedGrid');
            this.solution = null;
            this.shareButton.hidden = true;
        }
    }

    displayDetectedGrid(base64Image) {
        this.detectedImage.src = 'data:image/png;base64,' + base64Image;
        this.detectedCard.style.display = 'block';
    }

    displayEditableGrid(grid) {
        const size = grid.length;
        this.size = size;
        const container = document.getElementById('originalGrid');
        layoutGrid(container, size);
        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const input = document.createElement('input');
                input.className = 'sudoku-cell given';
                boxClasses(input, row, col, size);
                input.inputMode = size === 9 ? 'numeric' : 'text';
                input.autocapitalize = 'characters';
                input.maxLength = 1;
                input.setAttribute('aria-label', t('cell', { row: row + 1, col: col + 1 }));
                input.value = symbol(grid[row][col]);
                input.addEventListener('focus', () => input.select());
                input.addEventListener('input', () => {
                    const value = parseSymbol(input.value.slice(-1), size);
                    input.value = symbol(value);
                    input.classList.toggle('edited', !this.manual && value !== grid[row][col]);
                    this.resolveButton.disabled = false;
                    if (input.value && this.manual) this.focusCell(row * size + col + 1);
                });
                input.addEventListener('keydown', (e) => {
                    const step = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -size, ArrowDown: size }[e.key];
                    if (step) {
                        e.preventDefault();
                        this.focusCell(row * size + col + step);
                    } else if (e.key === 'Backspace' && !input.value) {
                        this.focusCell(row * size + col - 1);
                    } else if (e.key === 'Enter') {
                        this.solveCorrected();
                    }
                });
                container.appendChild(input);
            }
        }
        this.resolveButton.disabled = true;
    }

    focusCell(index) {
        const inputs = document.querySelectorAll('#originalGrid input');
        // preventScroll: focusing a cell would otherwise scroll the grid, which clips overflow
        if (index >= 0 && index < inputs.length) inputs[index].focus({ preventScroll: true });
    }

    readEditableGrid() {
        const inputs = document.querySelectorAll('#originalGrid input');
        const size = this.size;
        const grid = [];
        for (let i = 0; i < size * size; i++) {
            const text = inputs[i].value;
            const value = text === '' ? 0 : parseSymbol(text, size);
            if (text !== '' && value === 0) {
                return null;
            }
            if (i % size === 0) grid.push([]);
            grid[grid.length - 1].push(value);
        }
        return grid;
    }

    displayGrid(gridData, elementId, givens = null) {
        const size = gridData ? gridData.length : this.size;
        const container = document.getElementById(elementId);
        layoutGrid(container, size);
        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const cell = document.createElement('div');
                cell.className = 'sudoku-cell';
                boxClasses(cell, row, col, size);
                const value = gridData ? gridData[row][col] : 0;
                if (value !== 0) {
                    cell.textContent = symbol(value);
                    cell.classList.add(givens && givens[row][col] ? 'given' : 'solved');
                }
                container.appendChild(cell);
            }
        }
    }

    // The solution as an image, shared through the system share sheet where
    // the browser supports sharing files, downloaded otherwise.
    async shareSolution() {
        if (!this.solution) return;
        const blob = await new Promise((resolve) => this.drawSolution().toBlob(resolve, 'image/png'));
        const file = new File([blob], t('shareFile'), { type: 'image/png' });
        const url = location.origin + location.pathname;
        if (navigator.canShare && navigator.canShare({ files: [file] })) {
            try {
                await navigator.share({ files: [file], text: t('shareText') + ' ' + url });
            } catch {
                // cancelled
            }
            return;
        }
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = file.name;
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }

    drawSolution() {
        const { givens, solved } = this.solution;
        const size = solved.length, box = boxSide(size);
        const pad = 60, top = 140, side = 900, cell = side / size;
        const canvas = document.createElement('canvas');
        canvas.width = side + pad * 2;
        canvas.height = top + side + 130;
        const ctx = canvas.getContext('2d');
        const font = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#1e293b';
        ctx.font = `700 56px ${font}`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(t('brand'), canvas.width / 2, top / 2 + 10);
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(pad, top, side, side);
        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const x = pad + col * cell, y = top + row * cell;
                if (givens[row][col]) {
                    ctx.fillStyle = '#f1f5f9';
                    ctx.fillRect(x, y, cell, cell);
                }
                ctx.fillStyle = givens[row][col] ? '#1e293b' : '#3b82f6';
                ctx.font = `${givens[row][col] ? 700 : 500} ${Math.round(cell * 0.6)}px ${font}`;
                ctx.fillText(symbol(solved[row][col]), x + cell / 2, y + cell / 2 + 3);
            }
        }
        for (let i = 0; i <= size; i++) {
            ctx.strokeStyle = i % box === 0 ? '#334155' : '#94a3b8';
            ctx.lineWidth = i % box === 0 ? 6 : 2;
            ctx.beginPath();
            ctx.moveTo(pad + i * cell, top);
            ctx.lineTo(pad + i * cell, top + side);
            ctx.moveTo(pad, top + i * cell);
            ctx.lineTo(pad + side, top + i * cell);
            ctx.stroke();
        }
        ctx.fillStyle = '#64748b';
        ctx.font = `500 40px ${font}`;
        ctx.fillText(location.host + location.pathname.replace(/\/$/, ''), canvas.width / 2, top + side + 65);
        return canvas;
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
