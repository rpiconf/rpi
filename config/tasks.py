from config.constants import LOCAL_MODELS_PREFIX
from utils.dataclasses import Task

IMDB_TASK: Task = Task(  #
    dataset_name = "imdb",  #
    dataset_test = "test", dataset_column_text = "text",  #
    dataset_column_label = "label",  #
    bert_fine_tuned_model = "textattack/bert-base-uncased-imdb",  #
    roberta_fine_tuned_model = "textattack/roberta-base-imdb",  #
    distilbert_fine_tuned_model = "textattack/distilbert-base-uncased-imdb",  #
    labels_str_int_maps = dict(negative = 'N', positive = 'P'),  #
    test_sample = 2_000,  #
    name = "imdb",  #
    paper_name = "IMDB",  #
    is_llm_set_max_len = True,  #
    llm_explained_tokenizer_max_length = 400,  #
    llama_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf",  #
    mistral_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/mistralai_Mistral-7B-v0.1",  #
    is_finetuned_with_lora = False,  #
    llm_task_prompt = "Classify the sentiment of the movie review. For each sentence the label is positive (P) or negative (N)",
    llm_few_shots_prompt = [  #
        (
            "This movie is so bad, I knew how it ends right after this little girl killed the first person. Very bad acting very bad plot very bad movie<br /><br />do yourself a favour and DON'T watch it 1/10",
            #
            "N"),  #
        (
            "Very smart, sometimes shocking, I just love it. It shoved one more side of David's brilliant talent. He impressed me greatly! David is the best. The movie captivates your attention for every second.",
            #
            "P"), #
        (
            "If there is a movie to be called perfect then this is it. So bad it wasn't intended to be that way. But superb anyway... Go find it somewhere. Whatever you do... Do not miss it!!!",
            #
            "P"), #
        ("Long, boring, blasphemous. Never have I been so glad to see ending credits roll", "N")  #
    ])

EMOTION_TASK: Task = Task(dataset_name = "emotion",  #
                          dataset_test = "test",  #
                          dataset_column_text = "text",  #
                          dataset_column_label = "label",
                          bert_fine_tuned_model = "bhadresh-savani/bert-base-uncased-emotion",  #
                          roberta_fine_tuned_model = "bhadresh-savani/roberta-base-emotion",  #
                          distilbert_fine_tuned_model = "Rahmat82/DistilBERT-finetuned-on-emotion",  #
                          llama_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf",  #
                          mistral_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/mistralai_Mistral-7B-v0.1",  #
                          is_finetuned_with_lora = True,  #
                          llama_adapter = f"{LOCAL_MODELS_PREFIX}/TRAINED_MODELS/LLAMA_emotions_is_bf16_True_is_use_prompt_False",
                          #
                          mistral_adapter = f"{LOCAL_MODELS_PREFIX}/TRAINED_MODELS/MISTRAL_emotions_is_bf16_True_is_use_prompt_False",
                          #
                          labels_str_int_maps = dict(sadness = "A", joy = "B", love = "C", anger = "D", fear = "E", surprise = "F"),
                          #
                          test_sample = None,  #
                          name = "emotions",  #
                          paper_name = "EMR",
                          llm_task_prompt = "Classify the emotion expressed in each sentences. for each sentence the label is sadness (A) or joy (B) or love (C) or anger (D) or fear (4) or surprise (E)",
                          llm_few_shots_prompt = [])

SST_TASK: Task = Task(dataset_name = "sst2",  #
                      dataset_test = "test",  #
                      dataset_column_text = "sentence",  #
                      dataset_column_label = "label",  #
                      bert_fine_tuned_model = "textattack/bert-base-uncased-SST-2",  #
                      roberta_fine_tuned_model = "textattack/roberta-base-SST-2",  #
                      distilbert_fine_tuned_model = "distilbert-base-uncased-finetuned-sst-2-english",
                      llama_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf",
                      mistral_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/mistralai_Mistral-7B-v0.1",
                      is_finetuned_with_lora = False,  #
                      labels_str_int_maps = dict(negative = 'N', positive = 'P'),  #
                      test_sample = None,  #
                      name = "sst",  #
                      paper_name = "SST2",  #
                      llm_task_prompt = "Classify the sentiment of sentences. For each sentence the label is positive (P) or negative (N)",
                      llm_few_shots_prompt = [  #
                          ("hide new secretions from the parental units",  #
                           "N"),  #
                          ("the greatest musicians",  #
                           "P"),  #
                          ("are more deeply thought through than in most ` right-thinking ' films", "P"),  #
                          ("on the worst revenge-of-the-nerds clichés the filmmakers could dredge up", "N")  #
                      ])

AGN_TASK: Task = Task(dataset_name = "ag_news",  #
                      dataset_test = "test",  #
                      dataset_column_text = "text",  #
                      dataset_column_label = "label",  #
                      bert_fine_tuned_model = "fabriceyhc/bert-base-uncased-ag_news",
                      roberta_fine_tuned_model = "textattack/roberta-base-ag-news",
                      distilbert_fine_tuned_model = f"{LOCAL_MODELS_PREFIX}/TRAINED_MODELS/agn_distillbert",
                      labels_str_int_maps = dict(world = "A", sports = "B", business = "C", sci_tech = "D"),
                      test_sample = 2_000, name = "agn",  #
                      paper_name = "AGN",
                      llama_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf",
                      mistral_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/mistralai_Mistral-7B-v0.1",
                      llama_adapter = f"{LOCAL_MODELS_PREFIX}/TRAINED_MODELS/LLAMA_agn_is_bf16_True_is_use_prompt_False",
                      mistral_adapter = f"{LOCAL_MODELS_PREFIX}/TRAINED_MODELS/MISTRAL_agn_is_bf16_True_is_use_prompt_False",
                      is_finetuned_with_lora = True,
                      llm_task_prompt = "Classify the news articles. For each article label is World (A) Sports (B) Business (C) Sci/Tech (D)",
                      llm_few_shots_prompt = [])

RTN_TASK: Task = Task(dataset_name = "rotten_tomatoes",  #
                      dataset_test = "test",  #
                      dataset_column_text = "text",  #
                      dataset_column_label = "label",
                      bert_fine_tuned_model = "textattack/bert-base-uncased-rotten-tomatoes",
                      roberta_fine_tuned_model = "textattack/roberta-base-rotten-tomatoes",
                      distilbert_fine_tuned_model = "textattack/distilbert-base-uncased-rotten-tomatoes",
                      labels_str_int_maps = dict(negative = 'N', positive = 'P'),  #
                      test_sample = None,  #
                      name = "rtn",  #
                      paper_name = "RTN",  #
                      llama_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf",
                      mistral_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/mistralai_Mistral-7B-v0.1",
                      is_finetuned_with_lora = False,
                      llm_task_prompt = "Classify the sentiment of sentences. For each sentence the label is positive (P) or negative (N)",
                      llm_few_shots_prompt = [
                          ("the film desperately sinks further and further into comedy futility .",  #
                           #
                           "N"),  #
                          ("if you sometimes like to go to the movies to have fun , wasabi is a good place to start .",
                           "P"),  #
                          ("plays like the old disease-of-the-week small-screen melodramas .",  #
                           "N"),  #
                          ("hip-hop has a history , and it's a metaphor for this love story .",  #
                           "P"),  #
                          ("spiderman rocks",  #
                           "P"),  #
                          ("so exaggerated and broad that it comes off as annoying rather than charming .",  #
                           "N")  #
                      ])
