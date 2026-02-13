import random
import os
import sys
sys.path.append(os.getcwd())
import pyarrow as pa
import pyarrow.parquet as pq
from nanochat.common import get_base_dir

def generate_data():
    places = ['Nile', 'Pyramids', 'Khan El Khalili', 'Tahrir Square', 'Zamalek', 'Maadi']
    # Preferences with some consistency
    beverages = ['black coffee', 'tea', 'shai', 'turkish coffee', 'sahlab']
    foods = ['falafel', 'koshary', 'hawawshi', 'shawarma']
    
    lines = []
    
    # Simulation State
    current_place = random.choice(places)
    current_mood = "happy"
    favorite_drink = "tea" # User starts liking tea
    
    # Generate a coherent timeline from t=1 to t=10000
    for t in range(1, 10001):
        # 1. Occasionally change location (10% chance)
        if random.random() < 0.1:
            current_place = random.choice(places)
            
        # 2. Occasionally change preference (1% chance) - simulates evolving taste
        if random.random() < 0.01:
            favorite_drink = random.choice(beverages)
            
        # 3. Determine language for this turn
        lang = random.choice(['en', 'ar'])
        
        # 4. Generate Event
        if lang == 'en':
            if random.random() < 0.5:
                # Observation
                episode = f"t={t}: Visiting {current_place}, the atmosphere is {current_mood}. I ordered {favorite_drink}."
            else:
                # Action/Preference statement
                episode = f"t={t}: At {current_place} now. My favorite drink is currently {favorite_drink}."
        else:
            # Arabic equivalents (simplified mapping)
            place_ar = {
                'Nile': 'النيل', 'Pyramids': 'الأهرامات', 'Khan El Khalili': 'خان الخليلي',
                'Tahrir Square': 'ميدان التحرير', 'Zamalek': 'الزمالك', 'Maadi': 'المعادي'
            }.get(current_place, current_place)
            
            drink_ar = {
                'black coffee': 'قهوة سوداء', 'tea': 'شاي', 'shai': 'شاي', 
                'turkish coffee': 'قهوة تركي', 'sahlab': 'سحلب'
            }.get(favorite_drink, favorite_drink)
            
            if random.random() < 0.5:
                episode = f"t={t}: زيارة {place_ar}, الجو {current_mood}. طلبت {drink_ar}."
            else:
                episode = f"t={t}: أنا في {place_ar} الآن. مشروبي المفضل حالياً هو {drink_ar}."
                
        lines.append(episode)
    
    # Split into train (9000) and val (1000) - though for "Live Memory" we mostly care about the sequence
    train_lines = lines[:9000]
    val_lines = lines[9000:]
    
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "base_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save as Parquet (for training loader)
    train_table = pa.Table.from_pydict({"text": train_lines})
    pq.write_table(train_table, os.path.join(data_dir, "shard_00001.parquet"))
    
    val_table = pa.Table.from_pydict({"text": val_lines})
    pq.write_table(val_table, os.path.join(data_dir, "shard_00002.parquet"))
    
    # Save as Text (for ingestion script)
    txt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "synthetic-cairo-episodes.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        
    print(f"Generated {len(lines)} coherent episodes.")
    print(f"Text file: {txt_path}")
    print(f"Parquet shards in: {data_dir}")

if __name__ == "__main__":
    generate_data()
