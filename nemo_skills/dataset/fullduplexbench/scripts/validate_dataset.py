# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Validate Full-Duplex-Bench dataset and compute statistics.

This script:
1. Validates dataset integrity (files, format, audio)
2. Computes dataset statistics (counts, durations, etc.)
3. Checks for common issues
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available. Audio validation will be skipped.")


SUBTESTS = ["pause", "backchannel", "turn_taking", "interruption"]


def validate_subtest(data_dir: Path, subtest: str, check_audio: bool = True) -> Dict:
    """Validate a single subtest dataset."""
    subtest_dir = data_dir / subtest
    test_file = subtest_dir / "test.jsonl"
    
    results = {
        'subtest': subtest,
        'valid': True,
        'issues': [],
        'stats': {},
    }
    
    # Check if directory exists
    if not subtest_dir.exists():
        results['valid'] = False
        results['issues'].append(f"Directory not found: {subtest_dir}")
        return results
    
    # Check if test.jsonl exists
    if not test_file.exists():
        results['valid'] = False
        results['issues'].append(f"test.jsonl not found: {test_file}")
        return results
    
    # Load and validate entries
    entries = []
    line_num = 0
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                if line.strip():
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        results['issues'].append(f"JSON error at line {line_num}: {e}")
                        results['valid'] = False
    except Exception as e:
        results['valid'] = False
        results['issues'].append(f"Error reading test.jsonl: {e}")
        return results
    
    if not entries:
        results['valid'] = False
        results['issues'].append("No entries found in test.jsonl")
        return results
    
    # Validate entry format
    required_fields = ['problem', 'messages', 'messages_text', 'messages_text_audio']
    optional_fields = ['expected_answer', 'id', 'audio_path', 'category', 'duration']
    
    for idx, entry in enumerate(entries):
        # Check required fields
        for field in required_fields:
            if field not in entry:
                results['issues'].append(f"Entry {idx}: Missing required field '{field}'")
                results['valid'] = False
        
        # Validate messages format
        for msg_field in ['messages', 'messages_text', 'messages_text_audio']:
            if msg_field in entry:
                msgs = entry[msg_field]
                if not isinstance(msgs, list) or len(msgs) < 2:
                    results['issues'].append(
                        f"Entry {idx}: '{msg_field}' should be a list with at least 2 messages"
                    )
                    results['valid'] = False
                else:
                    # Check message structure
                    for msg in msgs:
                        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                            results['issues'].append(
                                f"Entry {idx}: Invalid message format in '{msg_field}'"
                            )
                            results['valid'] = False
                            break
    
    # Validate audio files
    audio_issues = []
    audio_durations = []
    
    if check_audio and SOUNDFILE_AVAILABLE:
        audio_dir = data_dir / "data"
        
        for idx, entry in enumerate(entries):
            if 'audio_path' in entry:
                audio_path = data_dir / entry['audio_path']
                
                if not audio_path.exists():
                    audio_issues.append(f"Entry {idx}: Audio file not found: {audio_path}")
                else:
                    try:
                        info = sf.info(str(audio_path))
                        audio_durations.append(info.duration)
                        
                        # Check for suspicious durations
                        if info.duration < 0.5:
                            audio_issues.append(f"Entry {idx}: Suspiciously short audio ({info.duration:.2f}s)")
                        elif info.duration > 300:
                            audio_issues.append(f"Entry {idx}: Very long audio ({info.duration:.2f}s)")
                        
                    except Exception as e:
                        audio_issues.append(f"Entry {idx}: Error reading audio: {e}")
    
    if audio_issues:
        results['issues'].extend(audio_issues[:10])  # Limit to first 10
        if len(audio_issues) > 10:
            results['issues'].append(f"... and {len(audio_issues) - 10} more audio issues")
    
    # Compute statistics
    results['stats'] = {
        'num_entries': len(entries),
        'has_expected_answer': sum(1 for e in entries if 'expected_answer' in e),
        'has_audio': sum(1 for e in entries if 'audio_path' in e),
    }
    
    if audio_durations:
        results['stats']['audio_stats'] = {
            'count': len(audio_durations),
            'total_duration': sum(audio_durations),
            'mean_duration': sum(audio_durations) / len(audio_durations),
            'min_duration': min(audio_durations),
            'max_duration': max(audio_durations),
        }
    
    # Count unique IDs if present
    ids = [e.get('id') for e in entries if 'id' in e]
    if ids:
        results['stats']['unique_ids'] = len(set(ids))
        if len(set(ids)) != len(ids):
            results['issues'].append("Duplicate IDs found")
    
    return results


def print_validation_results(all_results: List[Dict]):
    """Print formatted validation results."""
    print("\n" + "="*80)
    print("FULL-DUPLEX-BENCH DATASET VALIDATION")
    print("="*80)
    
    all_valid = True
    total_entries = 0
    total_duration = 0
    
    for result in all_results:
        subtest = result['subtest']
        valid = result['valid']
        issues = result['issues']
        stats = result['stats']
        
        print(f"\n{subtest.upper()}")
        print("-"*80)
        
        # Status
        if valid:
            print("✓ Status: VALID")
        else:
            print("✗ Status: INVALID")
            all_valid = False
        
        # Statistics
        if stats:
            print(f"  Entries: {stats.get('num_entries', 0)}")
            print(f"  With expected answer: {stats.get('has_expected_answer', 0)}")
            print(f"  With audio: {stats.get('has_audio', 0)}")
            
            if 'audio_stats' in stats:
                audio_stats = stats['audio_stats']
                print(f"  Audio files validated: {audio_stats['count']}")
                print(f"  Total audio duration: {audio_stats['total_duration']:.1f}s ({audio_stats['total_duration']/60:.1f}m)")
                print(f"  Mean duration: {audio_stats['mean_duration']:.2f}s")
                print(f"  Duration range: [{audio_stats['min_duration']:.2f}s, {audio_stats['max_duration']:.2f}s]")
                
                total_duration += audio_stats['total_duration']
            
            total_entries += stats.get('num_entries', 0)
        
        # Issues
        if issues:
            print(f"\n  Issues ({len(issues)}):")
            for issue in issues[:5]:  # Show first 5
                print(f"    - {issue}")
            if len(issues) > 5:
                print(f"    ... and {len(issues) - 5} more issues")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Status: {'✓ ALL VALID' if all_valid else '✗ VALIDATION FAILED'}")
    print(f"Total entries: {total_entries}")
    if total_duration > 0:
        print(f"Total audio duration: {total_duration:.1f}s ({total_duration/60:.1f}m / {total_duration/3600:.2f}h)")
    print("="*80 + "\n")
    
    return all_valid


def export_validation_report(all_results: List[Dict], output_file: Path):
    """Export validation report to JSON."""
    report = {
        'validation_results': all_results,
        'summary': {
            'total_entries': sum(r['stats'].get('num_entries', 0) for r in all_results),
            'all_valid': all(r['valid'] for r in all_results),
            'subtests_validated': len(all_results),
        }
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Validation report exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Full-Duplex-Bench dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all subtests
  python validate_dataset.py \\
    --data_dir /path/to/fullduplexbench
  
  # Validate specific subtests
  python validate_dataset.py \\
    --data_dir /path/to/fullduplexbench \\
    --subtests pause turn_taking
  
  # Skip audio validation (faster)
  python validate_dataset.py \\
    --data_dir /path/to/fullduplexbench \\
    --no_audio
  
  # Export report
  python validate_dataset.py \\
    --data_dir /path/to/fullduplexbench \\
    --output validation_report.json
        """
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to fullduplexbench data directory"
    )
    parser.add_argument(
        "--subtests",
        nargs="+",
        default=SUBTESTS,
        help="Subtests to validate (default: all)"
    )
    parser.add_argument(
        "--no_audio",
        action="store_true",
        help="Skip audio file validation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Export validation report to JSON"
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1
    
    print(f"Validating dataset in: {args.data_dir}")
    print(f"Subtests: {', '.join(args.subtests)}")
    print(f"Audio validation: {'disabled' if args.no_audio else 'enabled'}")
    
    # Validate each subtest
    all_results = []
    for subtest in args.subtests:
        result = validate_subtest(args.data_dir, subtest, check_audio=not args.no_audio)
        all_results.append(result)
    
    # Print results
    all_valid = print_validation_results(all_results)
    
    # Export report if requested
    if args.output:
        export_validation_report(all_results, args.output)
    
    # Exit with appropriate code
    return 0 if all_valid else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
