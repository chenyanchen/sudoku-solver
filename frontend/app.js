// Sudoku Solver Frontend Application

class SudokuSolver {
    constructor() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.previewSection = document.getElementById('previewSection');
        this.originalImage = document.getElementById('originalImage');
        this.detectedCard = document.getElementById('detectedCard');
        this.detectedImage = document.getElementById('detectedImage');
        this.solveButton = document.getElementById('solveButton');
        this.resultsSection = document.getElementById('resultsSection');
        this.messageSection = document.getElementById('messageSection');
        this.selectedFile = null;

        this.init();
    }

    init() {
        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Drag and drop
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

        // Solve button
        this.solveButton.addEventListener('click', () => {
            this.solvePuzzle();
        });
    }

    handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            this.showMessage('Please select an image file', 'error');
            return;
        }

        this.selectedFile = file;
        this.clearMessages();

        // Display preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.originalImage.src = e.target.result;
            this.previewSection.style.display = 'block';
            this.resultsSection.style.display = 'none';
            this.detectedCard.style.display = 'none';
            // Auto-solve after image selection to skip manual button click
            this.solvePuzzle();
        };
        reader.readAsDataURL(file);
    }

    async solvePuzzle() {
        if (!this.selectedFile) {
            this.showMessage('Please select an image first', 'error');
            return;
        }

        this.setLoading(true);
        this.clearMessages();

        try {
            const formData = new FormData();
            formData.append('image', this.selectedFile);

            const response = await fetch('/api/v1/sudoku:solveImage', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.showSuccess(data);
            } else {
                this.showMessage(data.message || 'Failed to solve puzzle', 'warning');
                if (data.detected_image) {
                    this.displayDetectedGrid(data.detected_image);
                }
                if (data.original_grid) {
                    this.displayGrid(data.original_grid, 'originalGrid', false);
                }
            }
        } catch (error) {
            this.showMessage('Error communicating with server: ' + error.message, 'error');
        } finally {
            this.setLoading(false);
        }
    }

    showSuccess(data) {
        this.showMessage(data.message || 'Puzzle solved successfully!', 'success');

        // Display detected grid if available
        if (data.detected_image) {
            this.displayDetectedGrid(data.detected_image);
        }

        // Display grids
        this.displayGrid(data.original_grid, 'originalGrid', false);
        this.displayGrid(data.solved_grid, 'solvedGrid', true);

        this.resultsSection.style.display = 'block';
    }

    displayDetectedGrid(base64Image) {
        this.detectedImage.src = 'data:image/png;base64,' + base64Image;
        this.detectedCard.style.display = 'block';
    }

    displayGrid(gridData, elementId, isSolved) {
        const container = document.getElementById(elementId);
        container.innerHTML = '';

        if (!gridData || !Array.isArray(gridData)) {
            return;
        }

        for (let row = 0; row < 9; row++) {
            for (let col = 0; col < 9; col++) {
                const cell = document.createElement('div');
                cell.className = 'sudoku-cell';

                const value = gridData[row][col];
                if (value !== 0) {
                    cell.textContent = value;
                    if (!isSolved) {
                        cell.classList.add('given');
                    } else {
                        cell.classList.add('solved');
                    }
                }

                container.appendChild(cell);
            }
        }
    }

    setLoading(loading) {
        const btnText = this.solveButton.querySelector('.btn-text');
        const btnLoader = this.solveButton.querySelector('.btn-loader');

        if (loading) {
            this.solveButton.disabled = true;
            btnText.style.display = 'none';
            btnLoader.style.display = 'inline-flex';
        } else {
            this.solveButton.disabled = false;
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
        }
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

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new SudokuSolver();
});
