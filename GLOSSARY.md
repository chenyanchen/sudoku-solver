# Sudoku Solver

Reads a Sudoku puzzle from an image and shows a solution; the uploader corrects any digit read wrongly.

## Puzzle

**Givens**:
The digits printed in the puzzle's cells as they appear in the image, whoever wrote them there.
_Avoid_: clues, numbers

**Solution**:
A full grid that keeps every given and repeats no digit in a row, column or box. A puzzle may have none, one, or several.

**Ambiguous puzzle**:
A puzzle whose givens have more than one solution. It is still solvable: any of its solutions answers it.
_Avoid_: multiple solutions error, unsolved

**Unsolvable puzzle**:
A puzzle whose givens have no solution.
_Avoid_: invalid (that is repeated digits only)

## Recognition

**Reading**:
The givens the site recognized in one upload, with whether they have a unique solution.
_Avoid_: result, recognition result

**Misread**:
A cell whose reading differs from the digit in the image (including a missed or invented given). Distinct from an ambiguous puzzle: a correct reading can be ambiguous, and a misread can still have a solution.
_Avoid_: recognition error used for "ambiguous"

**Unique reading**:
A reading whose givens have exactly one solution; the recognizer's evidence that it read every digit right. It decides whether to trust or correct a reading, never whether a solution is shown.
_Avoid_: solved (for this meaning)

**Correction**:
The givens as edited by the uploader for a reading.

**Solved upload**:
An uploaded image whose latest reading has a solution and was not corrected.

**Solve rate**:
The share of distinct uploaded images that are solved uploads.

## Training data

**Label**:
The givens of a training or test image, transcribed from the image. It has at least one solution; several are allowed when the puzzle itself is ambiguous.

**Grid corners**:
The four corners of a puzzle's 9×9 grid in an image, as reviewed by a person. Every training image needs them.
_Avoid_: predicted corners (a model's guess, never training data)

**Synthetic image**:
A generated puzzle image with known grid corners. Used only for training, never to choose or release a model.

**Release gate**:
The set of labeled images a model must read exactly before it can be released.

**Test set**:
Labeled images kept apart from training and from the release gate, used only to report how a model does on images it has never been tuned on.
_Avoid_: validation set (a slice of the training images)
