


def get_prompt_format():
    

    prompt_dictionary = {
    "person_university": {
        "english": "The college that {} attended was called",
        "chinese": "{} 就读的大学名叫",
        "japanese": "{} が通った大学の名前は",
        "korean": "{} 이 다녔던 대학의 이름은",
        "spanish": "La universidad a la que asistió {} se llama",
        "french": "L'université où {} a étudié s'appelle"
    },
    "country_currency": {
        "english": "The official currency of {} is called the",
        "chinese": "{}的官方货币是",
        "japanese": "{}の公式通貨は",
        "korean": "{}의 공식 화폐는",
        "spanish": "La moneda oficial de {} se llama",
        "french": "La monnaie officielle de {} s'appelle"
    },

    
    "book_language": {
        "english": "The language that {} was originally written in was",
        "chinese": "{}最初编写时使用的语言为",
        "japanese": "{}が最初に書かれた言語は",
        "korean": "{}가 원래 작성된 언어는",
        "spanish": "El idioma en el que {} fue escrito originalmente es",
        "french": ""
    },
    "animal_classification": {
    "english": "{} is biologically classified as a",
    "chinese": "{} 在生物学上被分类为一种",
    "japanese": "{} は生物学的に分類される",
    "korean": "{}의 생물학적 분류는",
    "spanish": "{} está clasificado biológicamente como un",
    "french": "{} est biologiquement classé comme un"
  }
    }   



    fact_recall_relation_index = {"book_language": {"korean": [-2]}}
    fact_recall_relation_words = {
    "person_university": {
        "english": ["college", "attended"],
        "chinese": ["大学", "就读"],
        "japanese": ["大学", "通った"],
        "korean": ["대학", "다녔던"],
        "spanish": ["universidad", "asistió"],
        "french": ["université", "étudié"]
    },
    "country_currency": {
        "english": ["currency"],
        "chinese": ["货币"],
        "japanese": ["通貨"],
        "korean": ["화폐"],
        "spanish": ["moneda"],
        "french": ["monnaie"]
    },
    "book_language": {
        "english": ["language", "written","original"],
        "chinese": ["语言", "编写"],
        "japanese": ["言語", "書かれた"],
        "korean": ["언어", "작성된"],
        "spanish": ["idioma", "escrito"],
        "french": ["langue", "écrit"]
    },
    "animal_classification": {
        "english": ["classified", "biologically"],
        "chinese": ["分类", "生物学"],
        "japanese": ["分類", "生物学的"],
        "korean": ["분류", "생물학적으로"],
        "spanish": ["clasificado", "biológicamente"],
        "french": ["classé", "biologiquement"]
    },
    "country_language": {
        "english": ["language"],
        "chinese": ["语言"],
        "japanese": ["公用語"],
        "korean": ["공용어"],
        "spanish": ["idioma"],
        "french": ["langue"]
    },
    "country_religion": {
        "english": ["religion", "practiced"],
        "chinese": ["宗教"],
        "japanese": ["宗教", "信仰"],
        "korean": ["종교", "믿는"],
        "spanish": ["religión", "practicada"],
        "french": ["religion", "pratiquée"]
    },
    "language_family": {
        "english": ["language", "family"],
        "chinese": ["语系"],
        "japanese": ["語族"],
        "korean": ["어족"],
        "spanish": ["lenguas", "familia"],
        "french": ["langues", "famille"]
    },
    "musician_country": {
        "english": ["birth", "country"],
        "chinese": ["出生", "国家"],
        "japanese": ["出身", "国"],
        "korean": ["출생", "국가"],
        "spanish": ["nacimiento", "país"],
        "french": ["naissance", "pays"]
    },
    "musician_instruments": {
        "english": ["instrument", "played"],
        "chinese": ["乐器", "演奏"],
        "japanese": ["楽器", "演奏"],
        "korean": ["악기", "연주"],
        "spanish": ["instrumento", "toca"],
        "french": ["instrument", "joué"]
    },
    "object_color": {
        "english": ["color"],
        "chinese": ["颜色"],
        "japanese": ["色"],
        "korean": ["색깔"],
        "spanish": ["color"],
        "french": ["couleur"]
    }
    }
    fact_recall_relation_words ={'person_university': {'english': ['college', 'university', 'attended', 'school', 'institution', 'institute', 'campus', 'collegiate', 'secondary', 'education', 'academic', 'prep', 'tutor', 'theology', 'grammar'], 'chinese': ['大学', '就读'], 'korean': ['대학', '다녔던'], 'japanese': ['大学', '通った'], 'spanish': ['universidad', 'asistió'], 'french': ['université', 'étudié']}, 'country_currency': {'english': ['currency', 'money', 'dollar', 'euro', 'euros', 'peso', 'yen', 'pound', 'franc', 'real', 'rand', 'yuan', 'rial', 'sterling', 'coin', 'cent'], 'chinese': ['货币'], 'korean': ['화폐'], 'japanese': ['通貨'], 'spanish': ['moneda'], 'french': ['monnaie']}, 'book_language': {'english': ['language', 'originally', 'dialect', 'bilingual', 'tongue', 'readable', 'original', 'translate', 'literature', 'poetry', 'fiction', 'prose', 'written'], 'chinese': ['语言', '编写'], 'korean': ['언어', '작성된'], 'japanese': ['言語', '書かれた'], 'spanish': ['idioma', 'escrito'], 'french': ['langue', 'écrit']}, 'animal_classification': {'english': ['classified', 'biological', 'classification', 'classify', 'taxonomy', 'taxa', 'scientific', 'organism', 'genus', 'species', 'order', 'dna'], 'chinese': ['分类', '生物学'], 'korean': ['분류', '생물학적으로'], 'japanese': ['分類', '生物学的'], 'spanish': ['clasificado', 'biológicamente'], 'french': ['classé', 'biologiquement']}, 'country_language': {'english': ['language', 'dialect', 'tongue', 'bilingual', 'native', 'speak', 'standard'], 'chinese': ['语言'], 'korean': ['공용어'], 'japanese': ['公用語'], 'spanish': ['idioma'], 'french': ['langue']}, 'country_religion': {'english': ['religion', 'religious', 'faith', 'belief', 'worship', 'practice', 'invocation', 'ethos', 'church'], 'chinese': ['宗教'], 'korean': ['종교', '믿는'], 'japanese': ['宗教', '信仰'], 'spanish': ['religión', 'practicada'], 'french': ['religion', 'pratiquée']}, 'language_family': {'english': ['language', 'family', 'dialect', 'tongue', 'kin', 'classification', 'subclass'], 'chinese': ['语系'], 'korean': ['어족'], 'japanese': ['語族'], 'spanish': ['lenguas', 'familia'], 'french': ['langues', 'famille']}, 'musician_country': {'english': ['country', 'birth', 'nation', 'state', 'locale', 'native', 'province', 'nationality', 'territory', 'homeland', 'locality'], 'chinese': ['出生', '国家'], 'korean': ['출생', '국가'], 'japanese': ['出身', '国'], 'spanish': ['nacimiento', 'país'], 'french': ['naissance', 'pays']}, 'musician_instruments': {'english': ['instrument', 'instrumental', 'instrumentation', 'played', 'playable', 'orchestra', 'solo'], 'chinese': ['乐器', '演奏'], 'korean': ['악기', '연주'], 'japanese': ['楽器', '演奏'], 'spanish': ['instrumento', 'toca'], 'french': ['instrument', 'joué']}, 'object_color': {'english': ['color', 'colour', 'hue', 'shade', 'tone', 'tint', 'pigment', 'paint'], 'chinese': ['颜色'], 'korean': ['색깔'], 'japanese': ['色'], 'spanish': ['color'], 'french': ['couleur']}}


    translate_to_english_prompt_dictionary = {
        "chinese": "请将这个词语翻译成英文。词语: ‘{}’; 翻译：‘",
        "korean": "이 단어를 영어로 번역해 주세요. 단어: ‘{}’; 번역: ‘",
        "japanese": "この言葉を英語に翻訳してください。言葉: ‘{}’; 翻訳: ‘",
        "spanish": "Por favor, traduce esta palabra al inglés. Palabra: ‘{}’; Traducción: ‘",
        "french": "Veuillez traduire ce mot en anglais. Mot : ‘{}’; Traduction : ‘",
        "english": "Please translate this word into English. Word: '{}'; Translation: '",
    }
    fv_icl_instruction_dict = {
        'animal_classification': {
            'chinese': '这种动物的生物分类是什么？',
            'korean': '이 동물의 생물학적 분류는 무엇인가요?',
            'japanese': 'この動物の分類は何ですか？',
            'french': "Quelle est la classification biologique de cet animal?",
            'spanish': "¿Cuál es la clasificación biológica de este animal?",
            'english': "What is the biological classification of this animal?"
        },
        'book_language': {
            'chinese': "这本书最初编写时使用的语言是什么？",
            'korean': "이 책이 처음 쓰인 언어는 무엇인가요?",
            'japanese': "この本が最初に書かれた言語は何ですか？",
            'french': "Dans quelle langue ce livre a-t-il été initialement écrit?",
            'spanish': "¿En qué idioma se escribió originalmente este libro?",
            'english': "What language was this book originally written in?"
        },
        'country_currency': {
            'chinese': '请告诉我每个国家所使用的货币。',
            'korean': '각 나라에서 사용하는 통화를 알려주세요.',
            'japanese': '各国で使われている通貨を教えてください。',
            'french': "Quelle est la monnaie utilisée dans chaque pays?",
            'spanish': "¿Cuál es la moneda utilizada en cada país?",
            'english': "What is the currency used in each country?"
        },
        'country_language': {
            'chinese': '这个国家的官方语言是什么？',
            'korean': '이 나라의 공식 언어는 무엇인가요?',
            'japanese': 'この国の公用語は何ですか？',
            'french': "Quelle est la langue officielle de ce pays?",
            'spanish': "¿Cuál es el idioma oficial de este país?",
            'english': "What is the official language of this country?"
        },
        'country_religion': {
            'chinese': '请告诉我每个国家的主要宗教。',
            'korean': '각 나라의 주요 종교를 알려주세요.',
            'japanese': '各国の主要な宗教を教えてください。',
            'french': "Quelle est la religion principale dans chaque pays?",
            'spanish': "¿Cuál es la religión principal en cada país?",
            'english': "What is the main religion in each country?"
        },
        'language_family': {
            'chinese': '这门语言属于哪个语系？',
            'korean': '이 언어는 어떤 어족에 속하나요?',
            'japanese': 'この言語はどの語族に属しますか？',
            'french': "À quelle famille linguistique appartient cette langue?",
            'spanish': "¿A qué familia lingüística pertenece este idioma?",
            'english': "Which language family does this language belong to?"
        },
        'musician_country': {
            'chinese': '请告诉我这个音乐家出生的国家。',
            'korean': '이 음악가가 태어난 나라는 어디인가요?',
            'japanese': 'この音楽家の出身国を教えてください。',
            'french': "Quel est le pays d'origine de ce musicien?",
            'spanish': "¿Cuál es el país de nacimiento de este músico?",
            'english': "What country is this musician from?"
        },
        'musician_instruments': {
            'chinese': '请告诉我这个音乐家演奏的乐器。',
            'korean': '이 음악가가 연주하는 악기를 알려주세요.',
            'japanese': 'この音楽家が演奏する楽器を教えてください。',
            'french': "Quels instruments joue ce musicien?",
            'spanish': "¿Qué instrumentos toca este músico?",
            'english': "What instruments does this musician play?"
        },
        'object_color': {
            'chinese': '这个东西是什么颜色的？',
            'korean': '이 물건은 무슨 색인가요?',
            'japanese': 'この物は何色ですか？',
            'french': "De quelle couleur est cet objet?",
            'spanish': "¿De qué color es este objeto?",
            'english': "What color is this object?"
        },
        'person_university': {
            'chinese': '这个人上过什么大学？',
            'korean': '이 사람이 다닌 대학은 어디인가요?',
            'japanese': 'この人が通った大学はどこですか？',
            'french': "Quelle université cette personne a-t-elle fréquentée?",
            'spanish': "¿A qué universidad asistió esta persona?",
            'english': "What university did this person attend?"
        }
    }
    translate_to_english_prompt_dictionary = {
        "chinese": "请将这个词语翻译成英文。词语: ‘{}’; 翻译：‘",
        "korean": "이 단어를 영어로 번역해 주세요. 단어: ‘{}’; 번역: ‘",
        "japanese": "この言葉を英語に翻訳してください。言葉: ‘{}’; 翻訳: ‘",
        "spanish": "Por favor, traduce esta palabra al inglés. Palabra: ‘{}’; Traducción: ‘",
        "french": "Veuillez traduire ce mot en anglais. Mot : ‘{}’; Traduction : ‘",
        "english": "Please translate this word into English. Word: ‘{}’; Translation：‘",
    }

    general_ICL_translate_to_english_prompt_dictionary = {
        "chinese": "请将这个词语翻译成英文。词语: ‘苹果’; 翻译：‘Apple’; 词语: ‘中国’; 翻译：‘China’; 词语: ‘罪与罚’; 翻译：‘Crime and Punishment’; 词语: ‘猪’; 翻译：‘Pig’; 词语: ‘艾米丽’; 翻译：‘Emily’; 词语: ‘{}’; 翻译：‘",
        "korean": "이 단어를 영어로 번역해 주세요. 단어: ‘사과’; 번역: ‘Apple’; 단어: ‘중국’; 번역: ‘China’; 단어: ‘죄와 벌’; 번역: ‘Crime and Punishment’; 단어: ‘돼지’; 번역: ‘Pig’; 단어: ‘에밀리’; 번역: ‘Emily’; 단어: ‘{}’; 번역: ‘",
        "japanese": "この言葉を英語に翻訳してください。言葉: ‘りんご’; 翻訳: ‘Apple’; 言葉: ‘中国’; 翻訳: ‘China’; 言葉: ‘罪と罰’; 翻訳: ‘Crime and Punishment’; 言葉: ‘豚’; 翻訳: ‘Pig’; 言葉: ‘エミリー’; 翻訳: ‘Emily’; 言葉: ‘{}’; 翻訳: ‘",
        "spanish": "Por favor, traduce esta palabra al inglés. Palabra: ‘manzana’; Traducción: ‘Apple’; Palabra: ‘China’; Traducción: ‘China’; Palabra: ‘Crimen y Castigo’; Traducción: ‘Crime and Punishment’; Palabra: ‘cerdo’; Traducción: ‘Pig’; Palabra: ‘Emily’; Traducción: ‘Emily’; Palabra: ‘{}’; Traducción: ‘",
        "french": "Veuillez traduire ce mot en anglais. Mot : ‘pomme’; Traduction : ‘Apple’; Mot : ‘Chine’; Traduction : ‘China’; Mot : ‘Crime et Châtiment’; Traduction : ‘Crime and Punishment’; Mot : ‘cochon’; Traduction : ‘Pig’; Mot : ‘Émilie’; Traduction : ‘Emily’; Mot : ‘{}’; Traduction : ‘",
        "english": "Please translate this word into English. Word: ‘Apple’; Translation: ‘Apple’; Word: ‘China’; Translation: ‘China’; Word: ‘Crime and Punishment’; Translation: ‘Crime and Punishment’; Word: ‘Pig’; Translation: ‘Pig’; Word: ‘Emily’; Translation: ‘Emily’; Word: ‘{}’; Translation: ‘",
    }

   
        
    return prompt_dictionary,translate_to_english_prompt_dictionary,fact_recall_relation_index,fact_recall_relation_words,fv_icl_instruction_dict,general_ICL_translate_to_english_prompt_dictionary



from prepare_ICL_fv import *
def new_load_ROME_data_for_task_dataset_language(TASK, DATASET, LANGUAGE, FROM_LAN=None,ICL_prompt=False,ICL_n_shots=None):
    
    #prompt_dictionary,translate_to_english_prompt_dictionary,fact_recall_relation_index,fact_recall_relation_words,fv_icl_instruction_dict,general_ICL_translate_to_english_prompt_dictionary = get_prompt_format()
    _,translate_to_english_prompt_dictionary,fact_recall_relation_index,fact_recall_relation_words,fv_icl_instruction_dict,general_ICL_translate_to_english_prompt_dictionary = get_prompt_format()
    file_path = None  # Initialize file path

    if TASK == 'fact_recall':
        fv_icl_prompt_template = {}
        if ICL_prompt:
            
            instruction = fv_icl_instruction_dict[DATASET][LANGUAGE]
            icl_data,zero_shot_data = construct_n_shot_zero_shot_datasets(DATASET,LANGUAGE,n_shot=ICL_n_shots,
                                                                      instruction = instruction)
            data = []
            for i in range(len(icl_data)):
                data.append({
                    "known_id": icl_data.known_ids[i],
                    "subject": icl_data.subjects[i],
                    "attribute": icl_data.completions[i],
                    "prompt": icl_data.prompts[i]
                })

        else:
            file_path = f"knowLLM_DATASETS/{TASK}/{DATASET}/{DATASET}_{LANGUAGE}.json"
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"Error: File not found: {file_path}")
                return None

        for entry in data:
            if DATASET in fact_recall_relation_words:
                entry["relation"] = fact_recall_relation_words[DATASET][LANGUAGE][0]
            #subject = entry.get("subject", "")
            if not ICL_prompt:
                #if DATASET in prompt_dictionary.keys() and len(prompt_dictionary[DATASET][LANGUAGE])>0:
                #    prompt_template = prompt_dictionary[DATASET][LANGUAGE]
                #    entry["prompt"] = prompt_template.format(subject)  # Correct format application
                if DATASET in fact_recall_relation_index.keys():
                    if LANGUAGE in fact_recall_relation_index[DATASET].keys():
                        if len(fact_recall_relation_index[DATASET][LANGUAGE])>0:
                            entry["relation_index"] = fact_recall_relation_index[DATASET][LANGUAGE] # Correct form
    elif TASK == 'translation':
        if FROM_LAN is None:
            print("TASK = translation, so NEED to provide FROM_LAN!")
            return None
        else:   
            with open(f"knowLLM_DATASETS/{TASK}/{DATASET}/from_{FROM_LAN}_to_{LANGUAGE}.json", "r") as f:
                data = json.load(f)
                for entry in data:
                    entry["subject"] = entry["prompt"].split("Word: '")[1].split("'. Translated:")[0].strip()
    elif TASK =='translate_to_english':
            with open(f"knowLLM_DATASETS/fact_recall/{DATASET}/{DATASET}_english.json", "r") as f:
                english_data = json.load(f)
           
            with open(f"knowLLM_DATASETS/fact_recall/{DATASET}/{DATASET}_{LANGUAGE}.json", "r") as f:
                data = json.load(f)
                for entry in data:
                    known_id = entry["known_id"]
                    subject = entry["subject"].strip().lower()
                    english_entry = next((e for e in english_data if e["known_id"] == known_id), None)
                    if not english_entry:
                        continue
                    english_subject = english_entry["subject"].strip().lower()
                    orig_word, trans_word = subject, english_subject
                    prompt_template = translate_to_english_prompt_dictionary[LANGUAGE]
                    entry["prompt"] = prompt_template.format(orig_word)  # Correct format application
                    #if ICL_prompt:
                    #    prompt_template = general_ICL_translate_to_english_prompt_dictionary[LANGUAGE]
                    #    entry["prompt"] = prompt_template.format(orig_word)  # Correct format application 
                    entry["subject"] = orig_word
                    entry["attribute"] = trans_word
                    entry['english_prompt'] = prompt_template.format(trans_word) #this is for activation patching

    else:
        print(f"Unsupported TASK: {TASK}")
        return None
    return data



import os

from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

def load_hooked_transformer_model(model_name):#model
    if 'llama' in model_name:
        from huggingface_hub import login
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token, add_to_git_credential=False)
        
        import torch
        if '1b' in model_name:
            if 'instruct' in model_name:
                model_id = 'meta-llama/Llama-3.2-1B-Instruct'
            else:
                model_id ="meta-llama/Llama-3.2-1B"# "meta-llama/Meta-Llama-3-8B-Instruct"
        if '7b' in model_name:
            model_id ="meta-llama/Llama-2-7b-hf"
        elif '8b' in model_name:
            model_id ="meta-llama/Meta-Llama-3-8B-Instruct"   
        elif '3b' in model_name:
            if 'instruct' in model_name:
                model_id ='meta-llama/Llama-3.2-3B-Instruct'
            else:
                model_id ='meta-llama/Llama-3.2-3B'
        else:
            pass

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        #model = AutoModelForCausalLM.from_pretrained(
        #    model_id,
        #    torch_dtype=torch.bfloat16,
        #    device_map="auto",
        #)
        #hf_model = LlamaForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)

        model = HookedTransformer.from_pretrained(model_id, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False)

        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    elif 'Qwen' in model_name:
        if '1.5' in model_name:
            model_name = 'Qwen/Qwen2-1.5B-Instruct'
        elif '0.5' in model_name:
            model_name = 'Qwen/Qwen2-0.5'
        elif '3' in model_name:
            model_name = 'Qwen/Qwen2.5-3B'
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = HookedTransformer.from_pretrained(model_name, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False)

        model = model.to("cuda")# if torch.cuda.is_available() else "cpu")
    else:
        print("MODEL NAME not configured in our code")
    return model,tokenizer
