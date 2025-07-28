# Document Structure Extractor

An advanced machine learning + rule-based system for extracting titles and headers from PDF documents with dynamic decision-making capabilities.

## Key Features

### Enhanced Capabilities
- **Dynamic Title Detection**: Intelligently determines if a document has a title or not
- **Smart Header Grouping**: Automatically detects and groups multi-line headers
- **Hybrid ML + Rule-Based**: Combines machine learning with robust rule-based fallbacks
- **No Title Support**: Can correctly identify documents without titles
- **Confidence Scoring**: Provides confidence scores for all predictions
- **1000+ File Training**: Designed to train on your large dataset

### Technical Improvements
- **Advanced Feature Engineering**: 25+ features including multi-line indicators
- **Rule-Based Title Scoring**: Dynamic scoring system for title candidates
- **Smart Header Grouping**: Groups split headers using 7 different criteria
- **Imbalanced Data Handling**: Uses class weights for better performance
- **Robust Training Pipeline**: Handles various document types and edge cases

## Files Overview

- `ml-model.py` - Main hybrid extractor with CLI interface
- `train_model3.py` - Training pipeline script
- `test_sample.py` - Quick test script for verification

## Installation

Install required dependencies:

```bash
pip install numpy pandas scikit-learn joblib
```

## Quick Start

### 1. Test on Sample File (No Training Required)

Test the rule-based approach on your sample file:

```bash
python test_sample.py
```

This will test the model on `./pdf-data-extracted/sample.json` and show results.

### 2. Train the Full Model

Train on your 1000+ JSON files:

```bash
python train_model.py
```

Or using the main script directly:

```bash
python ml-model.py train --training-data ./pdf-data-extracted --model extract-structure-data-model.joblib --debug
```

### 3. Predict on New Documents

Single document:
```bash
python ml-model.py predict --input document.json --output result.json --model extract-structure-data-model.joblib
```

Batch processing:
```bash
python ml-model.py batch --input ./input_folder --output ./results --model extract-structure-data-model.joblib
```

## Usage Examples

### Command Line Interface

```bash
# Train the model
python ml-model.py train --training-data ./pdf-data-extracted --model my_model.joblib

# Predict with debug info
python ml-model.py predict --input sample.json --output result.json --model my_model.joblib --debug

# Batch process multiple files
python ml-model.py batch --input ./docs --output ./results --model my_model.joblib

# Evaluate model performance
python ml-model.py evaluate --input ./test_data --model my_model.joblib
```

### Expected Output Format

```json
{
  "title": "THIS IS THE HEADING",
  "outline": [
    {
      "text": "Section 1: Introduction",
      "level": "H1",
      "page": 1,
    },
    {
      "text": "1.1 Background",
      "level": "H2", 
      "page": 1,
    }
  ],
}
```

## How It Works

### 1. Smart Header Grouping
- Groups consecutive items that belong together (multi-line headers)
- Uses font similarity, position, spacing, and formatting criteria
- Handles hyphenation, continuation patterns, and split titles

### 2. Dynamic Title Detection
- Rule-based scoring using document statistics
- Considers font size percentiles, position, formatting, and length
- Can determine "no title" when confidence is below threshold
- Dynamic thresholds based on document characteristics

### 3. ML-Enhanced Header Classification
- Extracts 25+ features including multi-line indicators
- Uses Gradient Boosting for title classification
- Uses Random Forest for heading level classification
- Handles imbalanced data with class weights

### 4. Hybrid Decision Making
- Falls back to rule-based approach when ML model unavailable
- Combines rule-based title detection with ML heading classification
- Provides confidence scores for all predictions

## 🎛️ Key Improvements Over Previous Versions

### Multi-Line Header Support
- Detects headers split across 2-3 lines
- Handles hyphenation, colons, and continuation patterns
- Smart grouping prevents incorrect merging

### Dynamic Title Detection
- No hardcoded assumptions about title presence
- Uses document-relative statistics for scoring
- Handles various document types (academic, business, technical)

### Better Training Data Handling
- Processes 1000+ files efficiently
- Creates training labels using hybrid heuristics
- Handles missing or inconsistent data gracefully

### Enhanced Feature Engineering
- Position-based features (page, y-position)
- Font-relative features (percentiles, ratios)
- Multi-line group features
- Content pattern features (numbering, capitalization)

## 🔧 Configuration Options

### Model Parameters
- `title_threshold`: Minimum confidence for title detection (default: 0.3)
- `heading_threshold`: Minimum confidence for heading detection (default: 0.2)
- `validation_split`: Training/validation split ratio (default: 0.2)

### Grouping Parameters
- Maximum group size: 4 items
- Font similarity tolerance: 20%
- Position tolerance: 10% of page height
- Maximum spacing threshold: 30 points

## Performance Characteristics

### Training Data
- Designed for 1000+ JSON files
- Handles various document types
- Creates balanced training labels automatically

### Prediction Speed
- Fast rule-based fallback
- Efficient feature extraction
- Batch processing support

### Accuracy Improvements
- Better title detection through dynamic scoring
- Improved header classification with enhanced features
- Robust handling of edge cases and unusual documents

## Troubleshooting

### Common Issues

1. **No title detected**: This is often correct! Many documents don't have clear titles
2. **Training fails**: Check that JSON files contain valid PDF extraction data
3. **Low confidence scores**: Normal for ambiguous documents - the model is being conservative

### Debug Mode
Use `--debug` flag to see detailed processing information:
```bash
python ml-model.py predict --input file.json --debug
```

### Validation
Test the model on known documents to verify performance:
```bash
python test_sample.py
```

## Notes

- The model can work without training (rule-based mode)
- Training improves performance but isn't always necessary
- Designed specifically for PDF text extraction JSON format
- Handles documents with no titles gracefully
- Provides confidence scores for better decision making


## Tips for Best Results

1. **Training Data**: Use diverse document types in training
2. **Validation**: Test on documents similar to your use case
3. **Thresholds**: Adjust confidence thresholds based on your requirements
4. **Debug Mode**: Use debug mode to understand model decisions
5. **Batch Processing**: Process multiple files for efficiency
