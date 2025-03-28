# textile-image-defect-detector
Proyecto de detección de defectos en productos textiles


Data Preprocessing Pipeline
This script handles data augmentation and train/test splitting for your image dataset.

1. Project Structure
your_project/
├── data/
│   ├── raw/               # Original images
│   │   ├── OK/            # 250 OK images
│   │   └── NO_OK/         # 90 NO_OK images
│   ├── augmented/         # Generated augmented images
│   └── split/            # Generated train/test splits
└── scripts/
    └── data_preprocessing.py
2. Setup
Place your original images in:

data/raw/OK/

data/raw/NO_OK/

Install required libraries:

```bash
pip install pillow numpy scikit-learn
```

3. Run the Script
Execute from the project root:

```bash
python scripts/data_preprocessing.py
```

4. Output
Augmented Images (Saved to data/augmented/):

OK/: ~750 images (250 × 3 augmentations)

NO_OK/: ~270 images (90 × 3 augmentations)

Train/Test Splits (Saved to data/split/):

train/OK/, train/NO_OK/: 80% of augmented data

test/OK/, test/NO_OK/: 20% of augmented data

5. Augmentation Techniques Applied
Technique	Parameter
Rotation	±30 degrees
Width/Height Shift	20% of image dimensions
Brightness	±30% adjustment
Zoom	20% range
Horizontal/Vertical Flip	Random
6. Customization
Edit these variables in data_preprocessing.py:

```python
AUGMENT_FACTOR = 3  # Generate 3x augmented images per original
TEST_SIZE = 0.2     # 20% for test set
SEED = 42           # For reproducible splits
```

7. Next Steps
Use these paths in your model training:

```python
train_dir = "data/split/train"
test_dir = "data/split/test"
```