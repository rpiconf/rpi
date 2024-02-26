from utils.dataclasses import Task

IMDB_TASK: Task = Task(dataset_name = "imdb", dataset_train = "train", dataset_val = "train", dataset_test = "test",
                       dataset_column_text = "text", dataset_column_label = "label",
                       bert_fine_tuned_model = "textattack/bert-base-uncased-imdb",
                       roberta_fine_tuned_model = "textattack/roberta-base-imdb",
                       roberta_base_model = "FacebookAI/roberta-base", bert_base_model = "bert-base-uncased",
                       labels_str_int_maps = dict(negative = 0, positive = 1), test_sample = 2_000, name = "imdb",
                       is_llama_set_max_len = True, llama_explained_tokenizer_max_length = 400,
                       llama_model = "/tmp/meta-llama_Llama-2-7b-hf",
                       llama_task_prompt = "Classify the sentiment of the movie review. For each sentence the label is positive (1) or negative (0)",
                       llama_few_shots_prompt = [  #
                           (
                               "This movie is so bad, I knew how it ends right after this little girl killed the first person. Very bad acting very bad plot very bad movie<br /><br />do yourself a favour and DON'T watch it 1/10",
                               #
                               0),  #
                           (
                               "Very smart, sometimes shocking, I just love it. It shoved one more side of David's brilliant talent. He impressed me greatly! David is the best. The movie captivates your attention for every second.",
                               #
                               1), (
                               "If there is a movie to be called perfect then this is it. So bad it wasn't intended to be that way. But superb anyway... Go find it somewhere. Whatever you do... Do not miss it!!!",
                               #
                               1),
                           ("Long, boring, blasphemous. Never have I been so glad to see ending credits roll", 0)  #
                       ])

EMOTION_TASK: Task = Task(dataset_name = "emotion", dataset_train = "train", dataset_val = "validation",
                          dataset_test = "test", dataset_column_text = "text", dataset_column_label = "label",
                          bert_fine_tuned_model = "bhadresh-savani/bert-base-uncased-emotion",
                          roberta_fine_tuned_model = "bhadresh-savani/roberta-base-emotion",
                          roberta_base_model = "FacebookAI/roberta-base", bert_base_model = "bert-base-uncased",
                          llama_model = "/tmp/meta-llama_Llama-2-7b-hf",
                          labels_str_int_maps = dict(sadness = 0, joy = 1, love = 2, anger = 3, fear = 4, surprise = 5),
                          test_sample = None, name = "emotions",
                          llama_task_prompt = "Classify the emotion expressed in each sentences. for each sentence the label is sadness (0) or joy (1) or love (2) or anger (3) or fear (4) or surprise (5)",
                          llama_few_shots_prompt = [('i feel when seeing a child suffering this way', 0),  #
                                                    ('im feeling a little overwhelmed here recently', 5),  #
                                                    ('i feel so annoyed', 3),  #
                                                    ('i do not feel overwhelmed nor rushed', 4),  #
                                                    ('i shook my head feeling dazed', 5),  #
                                                    ('i cant feel complacent', 1),  #
                                                    ('i mean post and i feel rotten abou', 0),  #
                                                    ('im feeling about as horny as a dead goat', 2),  #
                                                    ('i feel like an emotional train wreck', 0),  #
                                                    ('i feel pressured by a dumb feeling', 4),  #
                                                    ('i miss the feeling of feeling amazing', 5),  #
                                                    ('i feel so frustrated but i cant tell them i am', 3),  #
                                                    ('i feel fearful of being near them', 4),  #
                                                    ('i just feel like its rude', 3),  #
                                                    ('i didnt really feel like being thankful', 1),  #
                                                    ('i must feel loving toward everyone', 2),  #
                                                    ('i felt good and feel fine today too', 1),  #
                                                    ('i began to feel accepted by gaia on her own terms', 2)  #

                                                    ])

SST_TASK: Task = Task(dataset_name = "sst2", dataset_train = "train", dataset_val = "train",
                      dataset_test = "validation", dataset_column_text = "sentence", dataset_column_label = "label",
                      bert_fine_tuned_model = "textattack/bert-base-uncased-SST-2",
                      roberta_fine_tuned_model = "textattack/roberta-base-SST-2",
                      llama_model = "/tmp/meta-llama_Llama-2-7b-hf", labels_str_int_maps = dict(n = 0, p = 1),
                      roberta_base_model = "FacebookAI/roberta-base", bert_base_model = "bert-base-uncased",
                      test_sample = None, name = "sst",
                      llama_task_prompt = "Classify the sentiment of sentences. for each sentence the label is positive (1) or negative (0)",
                      llama_few_shots_prompt = [("hide new secretions from the parental units", 0),
                                                ("the greatest musicians", 1),  #
                                                (
                                                    "are more deeply thought through than in most ` right-thinking ' films",
                                                    1),  #
                                                (
                                                    "on the worst revenge-of-the-nerds clichés the filmmakers could dredge up",
                                                    0)  #
                                                ])

AGN_TASK: Task = Task(dataset_name = "ag_news", dataset_train = "train", dataset_val = "train", dataset_test = "test",
                      dataset_column_text = "text", dataset_column_label = "label",
                      bert_fine_tuned_model = "fabriceyhc/bert-base-uncased-ag_news",
                      roberta_fine_tuned_model = "textattack/roberta-base-ag-news",
                      labels_str_int_maps = dict(world = 0, sports = 1, business = 2, sci_tech = 3),
                      test_sample = 2_000, name = "agn", llama_model = "/tmp/meta-llama_Llama-2-7b-hf",
                      llama_task_prompt = "Classify the news articles. for each article label is World (0) Sports (1) Business (2) Sci/Tech (3)",
                      roberta_base_model = "FacebookAI/roberta-base", bert_base_model = "bert-base-uncased",
                      llama_few_shots_prompt = [(
                          'Bears Defeat Vikings, 24-14 (AP) AP - Hanging with Chad was a winning experience for the Chicago Bears.',
                          1),  #
                          (
                              "Dual-core IBM PowerPC 'to ship in single-core form' Standalone part - or a dualie with one dud core?",
                              3),  #
                          (
                              "Students Aiming to Avoid 'Freshman 15' AUSTIN - All-you-can-eat dorm dining. Late-night pizza parties...",
                              0),  #
                          (
                              'Henry Relishes Job of Covering Owens (AP) AP - No Sharpies. No situps. No pom-pom shaking. No spikes.',
                              1),  #
                          (
                              'Cricket: England whitewash England beat the West Indies by 10 wickets to seal a 4-0 series whitewash.',
                              0),  #
                          (
                              'Types of Investors: Which Are You? Learn a little about yourself, and it may improve your performance.',
                              2),  #
                          (
                              "Oracle's first monthly patch batch fails to placate critics Behind MS on security, says top bug hunter",
                              3),  #
                          (
                              "Pixar's Waiting for Summer Plus, IBM's win-win, Eli Lilly bares all, and a ticking retirement time bomb.",
                              2),  #
                          (
                              'BBC wants help developing open source video codec &lt;strong&gt;LinuxWorld&lt;/strong&gt; Dirac attack',
                              3),  #
                          (
                              'Baseball Today (AP) AP - Houston at St. Louis (8:19 p.m. EDT). Game 7 of the NL championship series.',
                              1),  #
                          (
                              'Poisoned. But Whodunit? The Ukrainian election takes a new twist after a stunning medical disclosure',
                              0),  #
                          (
                              "The iPod's Big Brother Apple's latest computer is as cool and sleek as its best-selling music player",
                              2)  #

                      ])

RTN_TASK: Task = Task(dataset_name = "rotten_tomatoes", dataset_train = "train", dataset_val = "validation",
                      dataset_test = "test", dataset_column_text = "text", dataset_column_label = "label",
                      bert_fine_tuned_model = "textattack/bert-base-uncased-rotten-tomatoes",
                      roberta_fine_tuned_model = "textattack/roberta-base-rotten-tomatoes",
                      roberta_base_model = "FacebookAI/roberta-base", bert_base_model = "bert-base-uncased",
                      labels_str_int_maps = dict(negative = 0, positive = 1), test_sample = None, name = "rtm",
                      llama_model = "/tmp/meta-llama_Llama-2-7b-hf",
                      llama_task_prompt = "Classify the sentiment of sentences. for each sentence the label is positive (1) or negative (0)",
                      llama_few_shots_prompt = [
                          ("the film desperately sinks further and further into comedy futility .",  #
                           0),  #
                          ("if you sometimes like to go to the movies to have fun , wasabi is a good place to start .",
                           1),  #
                          ("plays like the old disease-of-the-week small-screen melodramas .",  #
                           0),  #
                          ("hip-hop has a history , and it's a metaphor for this love story .",  #
                           1),  #
                          ("spiderman rocks",  #
                           1),  #
                          ("so exaggerated and broad that it comes off as annoying rather than charming .",  #
                           0)  #
                      ])
