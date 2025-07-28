"""
Environmental Integration Validation for SDS Reward Functions
============================================================
This module validates that generated reward functions properly integrate
environmental sensor data (height_scan and lidar_range).
"""

import re
import logging
from typing import Tuple, List, Dict

class EnvironmentalIntegrationValidator:
    """Validates environmental sensor integration in reward functions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Required patterns for environmental integration
        self.required_patterns = {
            'height_scan_access': [
                r'height_scan\s*=\s*env\.observation_manager\.get_term\(\s*["\']height_scan["\']\s*\)',
                r'height_scan.*=.*height_scan'  # Alternative access patterns
            ],
            'lidar_range_access': [
                r'lidar_range\s*=\s*env\.observation_manager\.get_term\(\s*["\']lidar_range["\']\s*\)',
                r'lidar_range.*=.*lidar_range'  # Alternative access patterns
            ],
            'terrain_analysis': [
                r'terrain_roughness.*torch\.var\(height_scan',
                r'local_terrain_level.*torch\.mean\(height_scan',
                r'terrain_complexity.*torch\.std\(height_scan',
                r'height_scan\.view\(.*env\.num_envs'
            ],
            'obstacle_detection': [
                r'forward_rays.*slice\(',
                r'min.*distance.*torch\.min\(lidar_range',
                r'obstacle.*distance',
                r'lidar_range\[:.*\]'
            ],
            'environmental_adaptation': [
                r'adaptive.*height',
                r'terrain.*adaptive',
                r'obstacle.*aware',
                r'gap.*navigation',
                r'stability.*requirement',
                r'distance_factor.*clamp',
                r'terrain.*complexity.*\*'
            ],
            'environmental_safety': [
                r'danger.*threshold',
                r'safety.*penalty',
                r'immediate_danger',
                r'hazard.*penalty',
                r'proximity.*penalty'
            ]
        }
        
        # Weight requirements
        self.weight_patterns = {
            'environmental_weighting': [
                r'env_weight',
                r'terrain.*weight',
                r'complexity.*\*.*\d+\.\d+',
                r'environmental.*weight'
            ]
        }
    
    def validate_reward_function(self, reward_code: str) -> Tuple[bool, Dict[str, any]]:
        """
        Validate a reward function for environmental integration.
        
        Args:
            reward_code: The reward function code as a string
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        validation_report = {
            'is_valid': True,
            'missing_components': [],
            'found_components': [],
            'score': 0,
            'recommendations': []
        }
        
        # Check each required pattern category
        for category, patterns in self.required_patterns.items():
            found = self._check_patterns(reward_code, patterns)
            if found:
                validation_report['found_components'].append(category)
                validation_report['score'] += 1
            else:
                validation_report['missing_components'].append(category)
                validation_report['is_valid'] = False
        
        # Check weighting patterns (bonus points)
        for category, patterns in self.weight_patterns.items():
            found = self._check_patterns(reward_code, patterns)
            if found:
                validation_report['found_components'].append(category)
                validation_report['score'] += 0.5
        
        # Generate recommendations
        validation_report['recommendations'] = self._generate_recommendations(
            validation_report['missing_components']
        )
        
        # Calculate final score (out of 6 required + 1 bonus)
        max_score = len(self.required_patterns)
        validation_report['score_percentage'] = (validation_report['score'] / max_score) * 100
        
        # Require at least 80% score for validation
        if validation_report['score_percentage'] < 80:
            validation_report['is_valid'] = False
        
        return validation_report['is_valid'], validation_report
    
    def _check_patterns(self, code: str, patterns: List[str]) -> bool:
        """Check if any of the patterns match in the code."""
        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    def _generate_recommendations(self, missing_components: List[str]) -> List[str]:
        """Generate specific recommendations for missing components."""
        recommendations = []
        
        if 'height_scan_access' in missing_components:
            recommendations.append(
                "Add height scan access: height_scan = env.observation_manager.get_term('height_scan')"
            )
        
        if 'lidar_range_access' in missing_components:
            recommendations.append(
                "Add LiDAR access: lidar_range = env.observation_manager.get_term('lidar_range')"
            )
        
        if 'terrain_analysis' in missing_components:
            recommendations.append(
                "Add terrain analysis: terrain_roughness = torch.var(height_scan.view(env.num_envs, -1), dim=1)"
            )
        
        if 'obstacle_detection' in missing_components:
            recommendations.append(
                "Add obstacle detection: min_distance = torch.min(lidar_range[:, forward_rays], dim=1)[0]"
            )
        
        if 'environmental_adaptation' in missing_components:
            recommendations.append(
                "Add environmental adaptation: adaptive behavior based on terrain complexity or obstacle proximity"
            )
        
        if 'environmental_safety' in missing_components:
            recommendations.append(
                "Add environmental safety: penalties for dangerous terrain or obstacle proximity"
            )
        
        return recommendations
    
    def log_validation_results(self, validation_report: Dict[str, any], reward_id: str = ""):
        """Log validation results."""
        prefix = f"Reward {reward_id}: " if reward_id else ""
        
        if validation_report['is_valid']:
            self.logger.info(
                f"{prefix}✅ Environmental integration PASSED "
                f"(Score: {validation_report['score_percentage']:.1f}%)"
            )
        else:
            self.logger.warning(
                f"{prefix}❌ Environmental integration FAILED "
                f"(Score: {validation_report['score_percentage']:.1f}%)"
            )
            
            for component in validation_report['missing_components']:
                self.logger.warning(f"  Missing: {component}")
            
            for rec in validation_report['recommendations']:
                self.logger.info(f"  Recommendation: {rec}")


def validate_reward_file(filepath: str) -> Tuple[bool, Dict[str, any]]:
    """
    Validate a reward function file for environmental integration.
    
    Args:
        filepath: Path to the reward function file
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    validator = EnvironmentalIntegrationValidator()
    
    try:
        with open(filepath, 'r') as f:
            reward_code = f.read()
        
        return validator.validate_reward_function(reward_code)
    
    except Exception as e:
        logging.error(f"Error validating reward file {filepath}: {e}")
        return False, {'error': str(e)}


if __name__ == "__main__":
    # Test validation on example files
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        is_valid, report = validate_reward_file(filepath)
        
        print(f"Validation Result: {'PASS' if is_valid else 'FAIL'}")
        print(f"Score: {report.get('score_percentage', 0):.1f}%")
        
        if not is_valid:
            print("\nMissing Components:")
            for component in report.get('missing_components', []):
                print(f"  - {component}")
            
            print("\nRecommendations:")
            for rec in report.get('recommendations', []):
                print(f"  - {rec}")
    else:
        print("Usage: python validate_environmental_integration.py <reward_file.py>") 