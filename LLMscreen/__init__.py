

def run(csv_file, filter_criteria, thread=16, api_file='api.txt', http_proxy='http://127.0.0.1:7890',
        https_proxy='http://127.0.0.1:7890', k=0, model="gpt-4o-mini-2024-07-18", zeroshot=False):
    import os
    import pandas as pd
    import numpy as np
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor
    import time
    # 计算用时
    start = time.time()
    """
    Process abstracts from a CSV file based on given filter criteria using OpenAI's GPT-3.5-turbo model.

    Parameters:
    - csv_file (str): Path to the CSV file containing the abstracts.
    - filter_criteria (str): The criteria for inclusion.
    - thread (int): Number of threads to use for parallel processing. Default is 16.
    - api_file (str): Path to the file containing the OpenAI API key. Default is 'api.txt'.
    - http_proxy (str): HTTP proxy URL. Default is 'http://127.0.0.1:7890'.
    - https_proxy (str): HTTPS proxy URL. Default is 'http://127.0.0.1:7890'.
    - k (float): The level of stringency for inclusion (used only in simple_mode).
    - model (str): The model to use (default is "gpt-4o-mini-2024-07-18").
    - zeroshot (bool): If True, use the zeroshot approach; otherwise, use simple_mode.

    Returns:
    - df_results (DataFrame): DataFrame containing the results of the abstract filtering.
    """

    def setup_openai_api(api_file='api.txt', http_proxy='http://127.0.0.1:7890', https_proxy='http://127.0.0.1:7890'):
        # Read API key from the file
        with open(api_file, 'r') as f:
            api = f.read().strip()

        # Set environment variables
        os.environ["OPENAI_API_KEY"] = api
        os.environ['http_proxy'] = http_proxy
        os.environ['https_proxy'] = https_proxy

        # Initialize and return OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api)

        return client

    def filter_abstract(client, title, abstract, filter_criteria, list_of_results, zeroshot=False, k=0.5,
                        model="gpt-4o-mini-2024-07-18"):
        if zeroshot:
            # Zeroshot approach
            question = ("Let's think step by step. First, think step by step about why inclusion might be true. "
                        "Second, think step by step about why exclusion might be true. "
                        "Third, think step by step about whether inclusion or exclusion makes more sense.")
            gpt_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are an academic assistant helping to evaluate studies for systematic reviews."},
                    {"role": "user",
                     "content": f'Title: {title}\nAbstract: {abstract} '
                                f'Way of thinking: {question}'
                                f'Give your chain of thought briefly without rushing to conclusions.'}
                ],
                temperature=0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_tokens=500
            ).choices[0].message.content

            # Filter abstract based on criteria
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are an academic who is currently doing a study on systematic reviews. Your task is to screen studies based on their titles and abstracts to determine if they meet the inclusion criteria."},
                    {"role": "system", "content": "Filter criteria: " + filter_criteria},
                    {"role": "system",
                     "content": "Give your judgement and answer True for inclusion and False for the opposite. (Answer only one word.)"},
                    {"role": "user",
                     "content": f'Here is the title and abstract text you will now be judged on: title[{title}]; abstract[{abstract}]'
                                f'Analysis:{gpt_response}'
                                f'Therefore, the answer is?'}
                ],
                temperature=0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logprobs=True
            )
        else:
            # Simple mode approach
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": "You are an academic who is currently doing a study on systematic reviews. Your task is to screen studies based on their titles and abstracts to determine if they meet the inclusion criteria." +
                                "The current level of stringency for inclusion is: k = " + str(
                         k) + ". (k is a number between 0 and 1, where 0 is the least stringent and 1 is the most stringent; When the parameter is 0 any paper can be included as long as it meets the meaning of the main body of the inclusion criteria, and when the parameter is 1 only papers that absolutely meet the criteria can be included.)"},
                    {"role": "system", "content": "filter criteria:" + filter_criteria},
                    {"role": "system",
                     "content": "Provide your decision as a JSON object with a field 'judgement' set to True (if the study should be included) or False (if the study should not be included), and a 'Reason' field briefly explains your decision to include or exclude the study."},
                    {"role": "user",
                     "content": f'Here is the title and abstract text you will now be judged on: title[{title}]; abstract[{abstract}]'},
                ],
                temperature=0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logprobs=True
            )

        # Common processing for both modes
        prob_content = response.choices[0].logprobs.content
        linear_probability = [np.exp(token.logprob) * 100 for token in prob_content]
        token_probability = np.mean([np.exp(token.logprob) * 100 for token in prob_content if
                                     token.token.lower().strip() == 'true' or token.token.lower().strip() == 'false'])
        n_probability = np.mean(linear_probability)
        logprobs = [token.logprob for token in prob_content]
        perplexity_score = np.exp(-np.mean(logprobs))

        list_of_results.append([response.choices[0].message.content, title, abstract,
                                n_probability, perplexity_score, token_probability])
        print("finished")

    # Setup OpenAI API
    client = setup_openai_api(api_file, http_proxy, https_proxy)

    # Load CSV file
    df = pd.read_csv(csv_file)

    # List to store results
    list_of_results = []
    # 计算用时
    start = time.time()
    # Use ThreadPoolExecutor to process abstracts in parallel
    with ThreadPoolExecutor(max_workers=thread) as executor:
        futures = [executor.submit(filter_abstract, client, row['title'], row['abstract'], filter_criteria,
                                   list_of_results, zeroshot, k, model) for index, row in df.iterrows()]
        for future in futures:
            future.result()

    df_results = pd.DataFrame(list_of_results,
                              columns=['judge', 'title', 'abstract', 'n_probability', 'perplexity_score',
                                       'token_probability'])
    #保存结果
    df_results.to_csv('result.csv', index=False)
    print("Done! All papers have been process")
    end = time.time()
    # 打印平均每一行用时
    print("The average time spent per paper was:", (end - start) / len(df))
    return df_results

