import json

def process_json_files(json1_file, json2_file):
    # Charger les fichiers JSON
    with open(json1_file, 'r') as file1:
        json1_data = json.load(file1)
    
    with open(json2_file, 'r') as file2:
        json2_data = json.load(file2)

    # Convertir le deuxième fichier JSON en dictionnaire pour un accès rapide
    json2_dict = {item['ID']: item for item in json2_data}

    # Parcourir le premier fichier JSON et appliquer les modifications nécessaires
    for obj in json1_data:
        obj_id = obj.get('ID')
        obj_type = obj.get('type', [])

        if obj_id in json2_dict:
            term = json2_dict[obj_id].get('term', '')
            sentence = json2_dict[obj_id].get('sentence', '')

            # Condition 1: Sentence non vide dans le fichier 2 et type "noun" ou "verb" dans le fichier 1 
            # if sentence and 'noun' in obj_type and len(obj_type) == 2:
            #     obj['type'] = ['verb']

            # Condition 1: Terme se terminant par 'ate' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('ate') and len(obj_type) == 2:
                obj['type'] = ['verb']
                
            # Condition 2: Terme se terminant par 'ify' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('ify') and len(obj_type) == 2:
                obj['type'] = ['verb']
                
            # Condition 3: Terme se terminant par 'ise' et ayant deux types dans le fichier 1
            # if len(term.split()) == 1 and term.endswith('ise') and len(obj_type) == 2:
            #     obj['type'] = ['verb']
                
            # Condition 4: Terme se terminant par 'ize' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('ize') and len(obj_type) == 2:
                obj['type'] = ['verb']
                
            # Condition 4: Terme se terminant par 'ible' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('ible') and len(obj_type) == 2:
                obj['type'] = ['adjective']
                
            # Condition 4: Terme se terminant par 'able' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('able') and len(obj_type) == 2:
                obj['type'] = ['adjective']
                
            # Condition 4: Terme se terminant par 'al' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('al') and len(obj_type) == 2:
                obj['type'] = ['adjective']
                
            # Condition 4: Terme se terminant par 'ic' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('ic') and len(obj_type) == 2:
                obj['type'] = ['adjective']
                
            # Condition 4: Terme se terminant par 'ous' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('ous') and len(obj_type) == 2:
                obj['type'] = ['adjective']
                
            # Condition 4: Terme se terminant par 'ful' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('ful') and len(obj_type) == 2:
                obj['type'] = ['adjective']
                
            # Condition 4: Terme se terminant par 'ive' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('ive') and len(obj_type) == 2:
                obj['type'] = ['adjective']
                
            # Condition 4: Terme se terminant par 'ous' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('ous') and len(obj_type) == 2:
                obj['type'] = ['adjective']
                
            # Condition 4: Terme se terminant par 'ly' et ayant deux types dans le fichier 1
            if len(term.split()) == 1 and term.endswith('ly') and len(obj_type) == 2:
                obj['type'] = ['adverb']

            # Condition 1: Terme se terminant par 'er' et ayant deux types dans le fichier 1
            # if len(term.split()) == 1 and term.endswith('er') and len(obj_type) == 2:
            #     obj['type'] = ['noun']
                
            # Condition 3: Terme composé de plusieurs mots
            # elif len(term.split()) > 1:
            #     obj['type'] = ['noun']
                
            # elif len(obj_type) == 2:
            #     obj['type'] = ['noun']
            
            # Tout les term qui sont seules et se terminent par tion et qui ont deux type

    # Sauvegarder le fichier 1 avec les modifications
    with open(json1_file, 'w') as outfile:
        json.dump(json1_data, outfile, indent=4)
    
    print("Process completed. The JSON file 1 has been updated.")

# Exemple d'utilisation
json1_file = '/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/tools/rulesapplies/datas/submission_taskA.json'  # Remplacer par le chemin de votre premier fichier JSON
json2_file = '/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/Test dataset/A.1(FS)_WordNet_Test.json'  # Remplacer par le chemin de votre second fichier JSON
process_json_files(json1_file, json2_file)
