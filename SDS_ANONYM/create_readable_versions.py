#!/usr/bin/env python3
"""
Utility script to create readable JSON versions for existing SDS runs
Usage: python3 create_readable_versions.py [specific_timestamp]
If no timestamp provided, processes all existing runs
"""
import json
import os
import sys
import glob

def create_readable_json_version(json_file_path):
    """Create a readable version of GPT query JSON files"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        readable_data = []
        for i, msg in enumerate(data):
            clean_msg = {'role': msg.get('role', 'unknown')}
            content = msg.get('content', '')
            
            if isinstance(content, list):
                clean_content = []
                for item in content:
                    if item.get('type') == 'text':
                        clean_content.append({
                            'type': 'text',
                            'text': item.get('text', '')
                        })
                    elif item.get('type') == 'image_url':
                        image_url = item.get('image_url', {})
                        if isinstance(image_url, dict) and 'url' in image_url:
                            if image_url['url'].startswith('data:image'):
                                clean_content.append({
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': '[BASE64_IMAGE_DATA_REMOVED_FOR_READABILITY]',
                                        'detail': image_url.get('detail', 'high'),
                                        'original_size_note': f'Original base64 data was {len(image_url["url"])} characters'
                                    }
                                })
                            else:
                                clean_content.append(item)
                        else:
                            clean_content.append({
                                'type': 'image_url',
                                'note': '[IMAGE_DATA_REMOVED_FOR_READABILITY]'
                            })
                    else:
                        clean_content.append(item)
                clean_msg['content'] = clean_content
            else:
                clean_msg['content'] = content
            
            readable_data.append(clean_msg)
        
        readable_path = json_file_path.replace('.json', '_READABLE.json')
        with open(readable_path, 'w', encoding='utf-8') as f:
            json.dump(readable_data, f, indent=2, ensure_ascii=False)
        
        original_size = os.path.getsize(json_file_path)
        readable_size = os.path.getsize(readable_path)
        
        print(f"✅ {os.path.basename(readable_path)} - {original_size/1024/1024:.1f}MB → {readable_size/1024:.1f}KB")
        return readable_path
        
    except Exception as e:
        print(f"❌ Failed: {json_file_path} - {e}")
        return None

def process_sds_run(run_dir):
    """Process all JSON files in an SDS run directory"""
    print(f"\n🔍 Processing: {os.path.basename(run_dir)}")
    
    # Find all JSON files that need readable versions
    json_files = []
    json_files.extend(glob.glob(os.path.join(run_dir, "reward_query_messages.json")))
    json_files.extend(glob.glob(os.path.join(run_dir, "evaluator_query_messages_*.json")))
    
    if not json_files:
        print("   ⚠️ No JSON files found")
        return
    
    for json_file in json_files:
        readable_path = json_file.replace('.json', '_READABLE.json')
        if os.path.exists(readable_path):
            print(f"   ⏭️ {os.path.basename(readable_path)} already exists")
        else:
            create_readable_json_version(json_file)

def main():
    sds_outputs_dir = "outputs/sds"
    
    if len(sys.argv) > 1:
        # Process specific timestamp
        timestamp = sys.argv[1]
        run_dir = os.path.join(sds_outputs_dir, timestamp)
        if os.path.exists(run_dir):
            process_sds_run(run_dir)
        else:
            print(f"❌ Directory not found: {run_dir}")
    else:
        # Process all existing runs
        print("🚀 Processing all existing SDS runs...")
        
        if not os.path.exists(sds_outputs_dir):
            print(f"❌ SDS outputs directory not found: {sds_outputs_dir}")
            return
        
        run_dirs = [d for d in os.listdir(sds_outputs_dir) 
                   if os.path.isdir(os.path.join(sds_outputs_dir, d))]
        run_dirs.sort()
        
        if not run_dirs:
            print("❌ No SDS runs found")
            return
        
        print(f"📊 Found {len(run_dirs)} SDS runs")
        
        for run_dir_name in run_dirs:
            run_dir = os.path.join(sds_outputs_dir, run_dir_name)
            process_sds_run(run_dir)
        
        print(f"\n🎯 Processed {len(run_dirs)} SDS runs")
        print("ℹ️ You can now easily view all GPT query data in GUI text editors!")

if __name__ == "__main__":
    main()
