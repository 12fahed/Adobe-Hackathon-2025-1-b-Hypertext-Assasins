#!/usr/bin/env python3
"""
Advanced Document Structure Extraction Model (v3)
Hybrid ML + Rule-Based Approach with Dynamic Title Detection

Features:
- Combines machine learning with rule-based methods
- Dynamic title detection with "no title" possibility
- Smart header grouping for multi-line elements
- Comprehensive feature engineering
- Robust training on 1000+ JSON files
- Real-time prediction with confidence scoring
"""

import json
import os
import sys
import logging
import argparse
import statistics
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Configure logging with UTF-8 encoding for Windows compatibility
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Remove emoji characters that cause encoding issues on Windows
            msg = msg.replace('✅', '[SUCCESS]').replace('❌', '[ERROR]').replace('🚀', '[START]').replace('📊', '[RESULTS]').replace('🎉', '[COMPLETE]')
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_extractor.log', encoding='utf-8'),
        UTF8StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RuleBasedTitleDetector:
    """Rule-based title detection engine with dynamic scoring"""
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def extract_title_candidates(self, grouped_items: List[List[Dict]], doc_stats: Dict) -> List[Dict]:
        """Extract and score title candidates using rule-based approach"""
        candidates = []
        
        for group in grouped_items:
            combined_text = " ".join([item['text'].strip() for item in group]).strip()
            
            if not combined_text or len(combined_text) < 3:
                continue
            
            # Calculate rule-based title score
            score = self._calculate_title_score_rules(group, combined_text, doc_stats)
            
            candidates.append({
                'text': combined_text,
                'score': score,
                'group': group,
                'page': group[0]['page'],
                'avg_font_size': statistics.mean([item['avg_font_size'] for item in group]),
                'y_position': min([item['y_position_relative'] for item in group]),
                'is_bold': any(item.get('is_bold', False) for item in group),
                'is_center': any(item.get('is_center_aligned', False) for item in group),
                'caps_ratio': statistics.mean([item.get('caps_ratio', 0) for item in group])
            })
        
        return sorted(candidates, key=lambda x: -x['score'])
    
    def _calculate_title_score_rules(self, group: List[Dict], text: str, doc_stats: Dict) -> float:
        """Calculate title score using rule-based features"""
        score = 0.0
        
        # Font size analysis (relative to document)
        avg_font_size = statistics.mean([item['avg_font_size'] for item in group])
        max_font_size = max([item['avg_font_size'] for item in group])
        
        # Score based on font size percentiles
        if max_font_size >= doc_stats.get('font_p95', 16):
            score += 8.0  # Top 5% font size
        elif max_font_size >= doc_stats.get('font_p85', 14):
            score += 6.0  # Top 15% font size
        elif max_font_size >= doc_stats.get('font_p75', 13):
            score += 4.0  # Top 25% font size
        
        # Relative to body text
        body_font = doc_stats.get('most_common_font', 12)
        font_ratio = avg_font_size / body_font if body_font > 0 else 1.0
        
        if font_ratio >= 2.0:
            score += 6.0
        elif font_ratio >= 1.5:
            score += 4.0
        elif font_ratio >= 1.3:
            score += 3.0
        elif font_ratio >= 1.1:
            score += 2.0
        
        # Position features - titles are typically early and high
        min_y_position = min([item['y_position_relative'] for item in group])
        page = group[0]['page']
        
        # Early page bonus
        if page == 1:
            score += 5.0
        elif page == 2:
            score += 2.0
        
        # High position on page
        if min_y_position <= 0.1:
            score += 6.0  # Top 10% of page
        elif min_y_position <= 0.2:
            score += 4.0  # Top 20% of page
        elif min_y_position <= 0.3:
            score += 2.0  # Top 30% of page
        
        # Formatting features
        bold_count = sum(1 for item in group if item.get('is_bold', False))
        center_count = sum(1 for item in group if item.get('is_center_aligned', False))
        
        if bold_count > 0 and doc_stats.get('bold_ratio', 0.5) < 0.3:
            score += 3.0
        
        if center_count > 0 and doc_stats.get('center_ratio', 0.5) < 0.2:
            score += 4.0
        
        # Text characteristics
        text_length = len(text)
        word_count = len(text.split())
        
        # Optimal title length
        if 5 <= word_count <= 12:
            score += 3.0
        elif 2 <= word_count <= 20:
            score += 2.0
        elif word_count > 30:
            score -= 4.0
        
        # Capitalization patterns
        caps_ratio = statistics.mean([item.get('caps_ratio', 0) for item in group])
        if caps_ratio > 0.7:  # Mostly uppercase
            score += 2.0
        elif caps_ratio > 0.3:  # Some uppercase
            score += 1.5
        
        # Spacing features
        avg_spacing = statistics.mean([item.get('spacing_above', 0) for item in group])
        if avg_spacing > doc_stats.get('avg_spacing', 10) * 2:
            score += 2.0
        
        # Penalty patterns
        if re.match(r'^\d+\s*$', text.strip()):  # Just numbers
            score -= 6.0
        if 'page' in text.lower() and word_count <= 3:
            score -= 5.0
        if len(text) > 200:  # Too long for title
            score -= 5.0
        if text.count('.') > 3:  # Too many sentences
            score -= 3.0
        
        return score


class SmartHeaderGrouper:
    """Advanced header grouping for multi-line elements"""
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def group_similar_items(self, data: List[Dict]) -> List[List[Dict]]:
        """Group consecutive items that belong together"""
        if not data:
            return []
        
        groups = []
        current_group = [data[0]]
        
        for i in range(1, len(data)):
            prev_item = current_group[-1]
            curr_item = data[i]
            
            if self._should_group_items(prev_item, curr_item, current_group):
                current_group.append(curr_item)
            else:
                groups.append(current_group)
                current_group = [curr_item]
        
        groups.append(current_group)
        
        if self.debug:
            multi_groups = [g for g in groups if len(g) > 1]
            logger.info(f"Created {len(multi_groups)} multi-line groups out of {len(groups)} total groups")
        
        return groups
    
    def _should_group_items(self, prev_item: Dict, curr_item: Dict, current_group: List[Dict]) -> bool:
        """Enhanced grouping logic with multiple criteria"""
        
        # Prevent overly long groups
        if len(current_group) >= 4:
            return False
        
        # Must be on same page
        if prev_item['page'] != curr_item['page']:
            return False
        
        # Font size similarity (20% tolerance)
        font_diff = abs(prev_item['avg_font_size'] - curr_item['avg_font_size'])
        font_threshold = max(prev_item['avg_font_size'], curr_item['avg_font_size']) * 0.2
        if font_diff > font_threshold:
            return False
        
        # Formatting compatibility
        format_matches = 0
        if prev_item.get('is_bold', False) == curr_item.get('is_bold', False):
            format_matches += 1
        if prev_item.get('is_italic', False) == curr_item.get('is_italic', False):
            format_matches += 1
        if prev_item.get('font_name', '') == curr_item.get('font_name', ''):
            format_matches += 1
        
        if format_matches < 2:
            return False
        
        # Vertical proximity (within 10% of page height)
        y_diff = abs(curr_item['y_position_relative'] - prev_item['y_position_relative'])
        if y_diff > 0.1:
            return False
        
        # Spacing threshold
        if curr_item.get('spacing_above', 0) > 30:
            return False
        
        # Text characteristics - avoid grouping long paragraphs
        total_length = sum(len(item['text']) for item in current_group) + len(curr_item['text'])
        if total_length > 300:
            return False
        
        # Multi-line indicators
        prev_text = prev_item['text'].strip()
        curr_text = curr_item['text'].strip()
        
        # Common continuation patterns
        if prev_text.endswith('-') or prev_text.endswith(':'):
            return True
        
        if curr_text.startswith(('and ', 'or ', 'the ', 'of ', 'in ', 'on ', 'with ')):
            return True
        
        # Both should be relatively short for grouping
        if len(prev_text) > 100 or len(curr_text) > 100:
            return False
        
        return True


class AdvancedFeatureExtractor:
    """Enhanced feature extraction with ML and rule-based features"""
    
    def extract_features(self, grouped_items: List[List[Dict]], doc_stats: Dict) -> pd.DataFrame:
        """Extract comprehensive features for ML training"""
        features = []
        
        for group in grouped_items:
            combined_text = " ".join([item['text'].strip() for item in group]).strip()
            
            if not combined_text:
                continue
            
            # Basic text features
            text_length = len(combined_text)
            word_count = len(combined_text.split())
            
            # Font features
            font_sizes = [item['avg_font_size'] for item in group]
            avg_font_size = statistics.mean(font_sizes)
            max_font_size = max(font_sizes)
            font_std = statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0
            
            # Position features
            y_positions = [item['y_position_relative'] for item in group]
            min_y_position = min(y_positions)
            avg_y_position = statistics.mean(y_positions)
            
            # Formatting features
            bold_ratio = sum(1 for item in group if item.get('is_bold', False)) / len(group)
            italic_ratio = sum(1 for item in group if item.get('is_italic', False)) / len(group)
            center_ratio = sum(1 for item in group if item.get('is_center_aligned', False)) / len(group)
            
            # Spacing features
            spacings = [item.get('spacing_above', 0) for item in group]
            avg_spacing = statistics.mean(spacings)
            max_spacing = max(spacings)
            
            # Capitalization features
            caps_ratios = [item.get('caps_ratio', 0) for item in group]
            avg_caps_ratio = statistics.mean(caps_ratios)
            
            # Indentation features
            indentations = [item.get('indentation_ratio', 0) for item in group]
            avg_indentation = statistics.mean(indentations)
            
            # Document-relative features
            font_ratio_to_body = avg_font_size / doc_stats.get('most_common_font', 12)
            font_percentile = self._calculate_percentile(avg_font_size, doc_stats.get('font_sizes', [12]))
            spacing_ratio = avg_spacing / max(doc_stats.get('avg_spacing', 1), 1)
            
            # Multi-line features
            group_size = len(group)
            is_grouped = group_size > 1
            has_hyphen_split = any(item['text'].strip().endswith('-') for item in group[:-1])
            has_colon_split = any(item['text'].strip().endswith(':') for item in group[:-1])
            
            # Pattern features
            has_numbering = bool(re.match(r'^\d+\.?\s+', combined_text))
            has_chapter_pattern = bool(re.search(r'chapter|section|part', combined_text, re.IGNORECASE))
            starts_with_capital = combined_text[0].isupper() if combined_text else False
            
            # Content analysis features
            sentence_count = combined_text.count('.') + combined_text.count('!') + combined_text.count('?')
            ends_with_period = combined_text.endswith('.')
            has_special_chars = bool(re.search(r'[^\w\s\.\,\-\:\;\!\?]', combined_text))
            
            feature_dict = {
                # Basic features
                'text_length': text_length,
                'word_count': word_count,
                'group_size': group_size,
                'page': group[0]['page'],
                
                # Font features
                'avg_font_size': avg_font_size,
                'max_font_size': max_font_size,
                'font_std': font_std,
                'font_ratio_to_body': font_ratio_to_body,
                'font_percentile': font_percentile,
                
                # Position features
                'min_y_position': min_y_position,
                'avg_y_position': avg_y_position,
                
                # Formatting features
                'bold_ratio': bold_ratio,
                'italic_ratio': italic_ratio,
                'center_ratio': center_ratio,
                'avg_caps_ratio': avg_caps_ratio,
                'avg_indentation': avg_indentation,
                
                # Spacing features
                'avg_spacing': avg_spacing,
                'max_spacing': max_spacing,
                'spacing_ratio': spacing_ratio,
                
                # Multi-line features
                'is_grouped': is_grouped,
                'has_hyphen_split': has_hyphen_split,
                'has_colon_split': has_colon_split,
                
                # Pattern features
                'has_numbering': has_numbering,
                'has_chapter_pattern': has_chapter_pattern,
                'starts_with_capital': starts_with_capital,
                
                # Content features
                'sentence_count': sentence_count,
                'ends_with_period': ends_with_period,
                'has_special_chars': has_special_chars,
                
                # Length categories
                'is_very_short': word_count <= 3,
                'is_short': word_count <= 10,
                'is_medium': 10 < word_count <= 30,
                'is_long': word_count > 30,
                
                # Position categories
                'is_top_of_page': min_y_position <= 0.1,
                'is_upper_half': min_y_position <= 0.5,
                'is_first_page': group[0]['page'] == 1,
                'is_early_page': group[0]['page'] <= 2,
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calculate percentile of value in list"""
        if not values:
            return 0.5
        sorted_values = sorted(values)
        rank = len([v for v in sorted_values if v <= value])
        return rank / len(sorted_values)


class HybridDocumentExtractor:
    """Hybrid ML + Rule-based document structure extractor"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.title_detector = RuleBasedTitleDetector()
        self.header_grouper = SmartHeaderGrouper()
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # ML components
        self.title_classifier = None
        self.heading_classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        
        # Model configuration
        self.title_threshold = 0.3  # Minimum confidence for title
        self.heading_threshold = 0.2  # Minimum confidence for heading
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _calculate_document_statistics(self, grouped_items: List[List[Dict]]) -> Dict:
        """Calculate comprehensive document statistics"""
        all_items = [item for group in grouped_items for item in group]
        
        if not all_items:
            return {}
        
        # Font analysis
        font_sizes = [item['avg_font_size'] for item in all_items]
        font_counts = Counter([round(size, 1) for size in font_sizes])
        most_common_font = font_counts.most_common(1)[0][0] if font_counts else 12.0
        
        # Spacing analysis
        spacings = [item.get('spacing_above', 0) for item in all_items if item.get('spacing_above', 0) > 0]
        
        # Calculate percentiles
        font_percentiles = np.percentile(font_sizes, [50, 75, 85, 95]) if font_sizes else [12, 12, 12, 12]
        
        # Formatting ratios
        bold_count = sum(1 for item in all_items if item.get('is_bold', False))
        center_count = sum(1 for item in all_items if item.get('is_center_aligned', False))
        
        return {
            'font_sizes': font_sizes,
            'most_common_font': most_common_font,
            'font_p50': font_percentiles[0],
            'font_p75': font_percentiles[1],
            'font_p85': font_percentiles[2],
            'font_p95': font_percentiles[3],
            'avg_spacing': statistics.mean(spacings) if spacings else 0,
            'bold_ratio': bold_count / len(all_items),
            'center_ratio': center_count / len(all_items),
            'total_items': len(all_items)
        }
    
    def train(self, training_data_folder: str, validation_split: float = 0.2):
        """Train the hybrid model on JSON files"""
        logger.info(f"Training on data from: {training_data_folder}")
        
        # Load training data
        training_documents = self._load_training_data(training_data_folder)
        
        if not training_documents:
            raise ValueError("No training data found!")
        
        logger.info(f"Loaded {len(training_documents)} training documents")
        
        # Process documents and extract features
        all_features = []
        all_title_labels = []
        all_heading_labels = []
        
        for doc_data in training_documents:
            # Group items
            grouped_items = self.header_grouper.group_similar_items(doc_data)
            doc_stats = self._calculate_document_statistics(grouped_items)
            
            # Extract features
            features_df = self.feature_extractor.extract_features(grouped_items, doc_stats)
            
            if len(features_df) == 0:
                continue
            
            # Create labels using hybrid approach
            title_labels, heading_labels = self._create_hybrid_labels(grouped_items, doc_stats)
            
            if len(title_labels) != len(features_df) or len(heading_labels) != len(features_df):
                continue
            
            all_features.append(features_df)
            all_title_labels.extend(title_labels)
            all_heading_labels.extend(heading_labels)
        
        if not all_features:
            raise ValueError("No valid features extracted from training data!")
        
        # Combine features
        X = pd.concat(all_features, ignore_index=True)
        y_title = np.array(all_title_labels)
        y_heading = np.array(all_heading_labels)
        
        logger.info(f"Training on {len(X)} samples")
        logger.info(f"Title distribution: {Counter(y_title)}")
        logger.info(f"Heading distribution: {Counter(y_heading)}")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        y_heading_encoded = self.label_encoder.fit_transform(y_heading)
        
        # Calculate class weights for imbalanced data
        title_weights = compute_class_weight('balanced', classes=np.unique(y_title), y=y_title)
        heading_weights = compute_class_weight('balanced', classes=np.unique(y_heading_encoded), y=y_heading_encoded)
        
        title_weight_dict = {i: w for i, w in enumerate(title_weights)}
        heading_weight_dict = {i: w for i, w in enumerate(heading_weights)}
        
        # Split data
        X_train, X_test, y_title_train, y_title_test, y_heading_train, y_heading_test = train_test_split(
            X_scaled, y_title, y_heading_encoded, test_size=validation_split, random_state=42
        )
        
        # Train title classifier
        logger.info("Training title classifier...")
        self.title_classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8
        )
        self.title_classifier.fit(X_train, y_title_train)
        
        # Train heading classifier
        logger.info("Training heading classifier...")
        self.heading_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            class_weight=heading_weight_dict,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.heading_classifier.fit(X_train, y_heading_train)
        
        # Evaluate models
        title_accuracy = accuracy_score(y_title_test, self.title_classifier.predict(X_test))
        heading_accuracy = accuracy_score(y_heading_test, self.heading_classifier.predict(X_test))
        
        logger.info(f"Title classifier accuracy: {title_accuracy:.3f}")
        logger.info(f"Heading classifier accuracy: {heading_accuracy:.3f}")
        
        # Feature importance
        if hasattr(self.title_classifier, 'feature_importances_'):
            title_importance = sorted(zip(self.feature_columns, self.title_classifier.feature_importances_), 
                                    key=lambda x: x[1], reverse=True)
            logger.info("Top 10 title features:")
            for feat, imp in title_importance[:10]:
                logger.info(f"  {feat}: {imp:.3f}")
        
        self.is_trained = True
        logger.info("Training completed successfully!")
    
    def _load_training_data(self, folder_path: str) -> List[List[Dict]]:
        """Load training data from JSON files"""
        documents = []
        folder = Path(folder_path)
        
        json_files = list(folder.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        documents.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        return documents
    
    def _create_hybrid_labels(self, grouped_items: List[List[Dict]], doc_stats: Dict) -> Tuple[List[int], List[str]]:
        """Create training labels using hybrid rule-based + heuristic approach"""
        title_labels = []
        heading_labels = []
        
        # Get rule-based title candidates
        title_candidates = self.title_detector.extract_title_candidates(grouped_items, doc_stats)
        
        # Determine title (best candidate above threshold)
        detected_title = None
        if title_candidates and title_candidates[0]['score'] > 6.0:
            detected_title = title_candidates[0]['text']
        
        for group in grouped_items:
            combined_text = " ".join([item['text'].strip() for item in group]).strip()
            
            # Title labeling
            is_title = (detected_title is not None and combined_text == detected_title)
            title_labels.append(1 if is_title else 0)
            
            # Heading labeling with rule-based heuristics
            heading_label = self._classify_heading_level(group, combined_text, doc_stats, is_title)
            heading_labels.append(heading_label)
        
        return title_labels, heading_labels
    
    def _classify_heading_level(self, group: List[Dict], text: str, doc_stats: Dict, is_title: bool) -> str:
        """Classify heading level using rule-based heuristics"""
        if is_title:
            return 'body'  # Title is not a heading
        
        # Basic filtering
        if len(text) < 3 or len(text) > 200:
            return 'body'
        
        word_count = len(text.split())
        if word_count > 25:
            return 'body'
        
        # Calculate heading score
        score = 0.0
        
        # Font size features
        avg_font_size = statistics.mean([item['avg_font_size'] for item in group])
        font_ratio = avg_font_size / doc_stats.get('most_common_font', 12)
        
        if font_ratio >= 1.5:
            score += 4.0
        elif font_ratio >= 1.3:
            score += 3.0
        elif font_ratio >= 1.1:
            score += 2.0
        
        # Formatting features
        if any(item.get('is_bold', False) for item in group):
            score += 2.0
        
        # Spacing features
        avg_spacing = statistics.mean([item.get('spacing_above', 0) for item in group])
        if avg_spacing > doc_stats.get('avg_spacing', 10) * 1.5:
            score += 2.0
        
        # Numbering patterns
        if re.match(r'^\d+\.?\s+', text):
            score += 2.0
            return 'H2'  # Numbered sections
        elif re.match(r'^\d+\.\d+\.?\s+', text):
            score += 2.5
            return 'H3'  # Sub-numbered sections
        
        # Position and length
        if group[0]['page'] == 1 and group[0]['y_position_relative'] < 0.3:
            score += 1.0
        
        if 3 <= word_count <= 10:
            score += 1.5
        
        # Classify based on score and features
        if score >= 4.0:
            if font_ratio >= 1.4:
                return 'H1'
            elif font_ratio >= 1.2:
                return 'H2'
            else:
                return 'H3'
        elif score >= 2.5:
            return 'H3'
        elif score >= 1.5:
            return 'H4'
        else:
            return 'body'
    
    def predict(self, json_data: List[Dict], debug: bool = False) -> Dict[str, Any]:
        """Predict document structure using hybrid approach"""
        if not json_data:
            return {"title": "", "outline": []}
        
        # Group items
        grouped_items = self.header_grouper.group_similar_items(json_data)
        doc_stats = self._calculate_document_statistics(grouped_items)
        
        if debug:
            logger.info(f"Grouped {len(json_data)} items into {len(grouped_items)} groups")
        
        # Phase 1: Rule-based title detection
        title_candidates = self.title_detector.extract_title_candidates(grouped_items, doc_stats)
        
        # Determine title with confidence
        detected_title = ""
        title_confidence = 0.0
        
        if title_candidates:
            best_candidate = title_candidates[0]
            title_confidence = min(best_candidate['score'] / 10.0, 1.0)  # Normalize to 0-1
            
            if best_candidate['score'] > 5.0:  # Dynamic threshold
                detected_title = best_candidate['text']
        
        if debug:
            logger.info(f"Title candidates: {len(title_candidates)}")
            if title_candidates:
                logger.info(f"Best title: '{detected_title}' (score: {title_candidates[0]['score']:.2f})")
        
        # Phase 2: ML-enhanced heading detection
        outline = []
        
        if self.is_trained:
            outline = self._predict_headings_ml(grouped_items, doc_stats, detected_title, debug)
        else:
            outline = self._predict_headings_rules(grouped_items, doc_stats, detected_title, debug)
        
        return {
            "title": detected_title,
            "outline": outline,
            # "title_confidence": title_confidence,
            # "total_groups": len(grouped_items),
            # "method": "hybrid_ml" if self.is_trained else "hybrid_rules"
        }
    
    def _predict_headings_ml(self, grouped_items: List[List[Dict]], doc_stats: Dict, 
                           detected_title: str, debug: bool) -> List[Dict]:
        """Predict headings using trained ML models"""
        # Extract features
        features_df = self.feature_extractor.extract_features(grouped_items, doc_stats)
        
        if len(features_df) == 0:
            return []
        
        # Ensure feature consistency
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[self.feature_columns]
        X_scaled = self.scaler.transform(features_df)
        
        # Predict
        title_probs = self.title_classifier.predict_proba(X_scaled)[:, 1]
        heading_predictions = self.heading_classifier.predict(X_scaled)
        heading_labels = self.label_encoder.inverse_transform(heading_predictions)
        heading_probs = self.heading_classifier.predict_proba(X_scaled)
        
        outline = []
        
        for i, group in enumerate(grouped_items):
            combined_text = " ".join([item['text'].strip() for item in group]).strip()
            
            # Skip if this is the detected title
            if combined_text == detected_title:
                continue
            
            # Check heading confidence
            heading_label = heading_labels[i]
            heading_prob = np.max(heading_probs[i])
            
            if heading_label != 'body' and heading_prob > self.heading_threshold:
                outline.append({
                    'text': combined_text,
                    'level': heading_label,
                    'page': group[0]['page'],
                    # 'confidence': float(heading_prob)
                })
        
        if debug:
            logger.info(f"ML predicted {len(outline)} headings")
        
        return outline
    
    def _predict_headings_rules(self, grouped_items: List[List[Dict]], doc_stats: Dict, 
                              detected_title: str, debug: bool) -> List[Dict]:
        """Predict headings using rule-based approach (fallback)"""
        outline = []
        
        for group in grouped_items:
            combined_text = " ".join([item['text'].strip() for item in group]).strip()
            
            if combined_text == detected_title or len(combined_text) < 3:
                continue
            
            heading_level = self._classify_heading_level(group, combined_text, doc_stats, False)
            
            if heading_level != 'body':
                outline.append({
                    'text': combined_text,
                    'level': heading_level,
                    'page': group[0]['page'],
                    'confidence': 0.7  # Default confidence for rule-based
                })
        
        if debug:
            logger.info(f"Rule-based predicted {len(outline)} headings")
        
        return outline
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        model_data = {
            'title_classifier': self.title_classifier,
            'heading_classifier': self.heading_classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'title_threshold': self.title_threshold,
            'heading_threshold': self.heading_threshold
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.title_classifier = model_data['title_classifier']
        self.heading_classifier = model_data['heading_classifier']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        self.title_threshold = model_data.get('title_threshold', 0.3)
        self.heading_threshold = model_data.get('heading_threshold', 0.2)
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Hybrid Document Structure Extractor v3")
    parser.add_argument('command', choices=['train', 'predict', 'batch', 'evaluate'], 
                       nargs='?', default='batch',
                       help='Command to execute (default: batch)')
    
    # Common arguments
    parser.add_argument('--model', '-m', type=str, default='extract-structure-data-model.joblib',
                       help='Path to model file')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    
    # Training arguments
    parser.add_argument('--training-data', '-t', type=str, default='./pdf-data-extracted',
                       help='Folder containing training JSON files')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Prediction arguments
    parser.add_argument('--input', '-i', type=str, help='Input JSON file or folder')
    parser.add_argument('--output', '-o', type=str, help='Output file or folder')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize extractor
    extractor = HybridDocumentExtractor()
    
    try:
        if args.command == 'train':
            logger.info("[START] Starting hybrid model training...")
            extractor.train(args.training_data, args.validation_split)
            extractor.save_model(args.model)
            logger.info(f"[SUCCESS] Training completed! Model saved to {args.model}")
        
        elif args.command == 'predict':
            if not args.input:
                raise ValueError("Input file required for prediction")
            
            # Load model
            if os.path.exists(args.model):
                extractor.load_model(args.model)
            
            # Load and process document
            with open(args.input, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            result = extractor.predict(json_data, args.debug)
            
            # Output result
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"[SUCCESS] Result saved to {args.output}")
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif args.command == 'batch':
            # if not args.input:
            #     raise ValueError("Input and output folders required for batch processing")
            
            # Load model
            if os.path.exists(args.model):
                extractor.load_model(args.model)
            
            # Process all JSON files
            input_folder = Path(args.input) if args.input else Path("./app/extracted-data")
            output_folder = Path(args.output) if args.output else Path("./app/output")
            output_folder.mkdir(exist_ok=True)
            
            json_files = list(input_folder.glob("*.json"))
            logger.info(f"Processing {len(json_files)} files...")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    result = extractor.predict(json_data, args.debug)
                    
                    output_file = output_folder / f"{json_file.stem}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"[SUCCESS] Processed {json_file.name}")
                
                except Exception as e:
                    logger.error(f"[ERROR] Failed to process {json_file.name}: {e}")
            
            logger.info("[COMPLETE] Batch processing completed!")
        
        elif args.command == 'evaluate':
            if not args.input:
                raise ValueError("Input folder required for evaluation")
            
            # Load model
            if os.path.exists(args.model):
                extractor.load_model(args.model)
            
            # Evaluate on test data
            test_files = list(Path(args.input).glob("*.json"))[:50]  # Sample evaluation
            
            title_found = 0
            heading_found = 0
            total_files = len(test_files)
            
            for json_file in test_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    result = extractor.predict(json_data, False)
                    
                    if result['title']:
                        title_found += 1
                    if result['outline']:
                        heading_found += 1
                
                except Exception as e:
                    logger.error(f"Error evaluating {json_file.name}: {e}")
            
            logger.info(f"[RESULTS] Evaluation Results on {total_files} files:")
            logger.info(f"   Files with titles: {title_found} ({title_found/total_files:.1%})")
            logger.info(f"   Files with headings: {heading_found} ({heading_found/total_files:.1%})")
    
    except Exception as e:
        logger.error(f"[ERROR] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Usage Examples:
"""
# Train the hybrid model
python ml-model.py train --training-data ./pdf-data-extracted --model extract-structure-data-model.joblib --debug

# Predict structure for a single document
python ml-model.py predict --input sample.json --output result.json --model extract-structure-data-model.joblib --debug

# Batch process multiple documents
python ml-model.py batch --input ./input_folder --output ./results --model extract-structure-data-model.joblib

# Evaluate model performance
python ml-model.py evaluate --input ./test_data --model extract-structure-data-model.joblib

# Train on specific subset of data
python ml-model.py train --training-data ./pdf-data-extracted --validation-split 0.15 --model specialized_model.joblib
"""
