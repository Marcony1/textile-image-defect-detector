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

### Label Studio Used Credentials
User: marcop.bravom@gmail.com
Password: galiatextil

### Alternative Instructions

## 📦 Data Extraction Utility

The `scripts/02_unzipper.py` script automatically extracts zip files into organized subdirectories.

### 🚀 Basic Usage
```bash
python scripts/02_unzipper.py path/to/yourfile.zip
```
📋 Examples
#### Extract from project's data directory
python scripts/unzip_data.py data/raw_measurements.zip

#### Extract from anywhere in your system
python scripts/unzip_data.py ~/downloads/archive.zip

### 🔄 Expected Behavior
```bash
Extracting survey_data.zip to ./survey_data
Successfully extracted to ./survey_data
```

## 🧩 YOLO Dataset Splitter (Stratified & Optional Test Set)
Split your YOLOv5/YOLOv8 dataset into train/valid/test folders with multilabel stratification support and automatic data.yaml creation.

### 📦 Folder Input Format:
```pgsql
dataset/
├── images/
├── labels/
├── classes.txt
```

### 🖥️ Usage
```bash
python scripts/03_data_splitter.py \
  --input data/01_annotated_data \
  --ratios 0.7 0.2 0.1 \
  --seed 42 \
  --stratify True
```

### 🧾 Output Structure
```pgsql
01_annotated_data_splitted/
├── train/
├── valid/
├── test/              ← Only if --ratios include test > 0
└── data.yaml          ← YOLO-compatible
```

### 📁 data.yaml Example
```yaml
train: train/images
val: valid/images
test: test/images        # only if test ratio was provided
nc: 5
names: ['cat', 'dog', 'car', 'tree', 'person']
```