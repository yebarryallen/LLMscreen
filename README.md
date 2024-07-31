

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![PyPI version](https://badge.fury.io/py/LLMscreen.svg)](https://badge.fury.io/py/LLMscreen)
[![Website](https://img.shields.io/badge/Website-Jinquan_Ye-red)](https://jinquanyescholar.netlify.app)


## Overview

`LLMscreen` is a package designed to filter and process research abstracts based on given criteria using OpenAI's language models. It supports both simple and zeroshot approaches for inclusion criteria and provides detailed outputs including probabilities and perplexity scores.

## Installation

To install the package, use the following command:

```bash
pip install LLMscreen
```


### Syntax

run(csv_file, filter_criteria, thread=16, api_file='api.txt', http_proxy='http://127.0.0.1:7890',
    https_proxy='http://127.0.0.1:7890', k=0, model="gpt-4o-mini-2024-07-18", zeroshot=False)


### Parameters
- **csv_file (str)**: Path to the CSV file containing the abstracts.
- **filter_criteria (str)**: The criteria for inclusion of abstracts.
- **thread (int)**: Number of threads for parallel processing (default is 16).
- **api_file (str)**: Path to the file containing the OpenAI API key (default is 'api.txt').
- **http_proxy (str)**: HTTP proxy URL (default is 'http://127.0.0.1:7890').
- **https_proxy (str)**: HTTPS proxy URL (default is 'http://127.0.0.1:7890').
- **k (float)**: Stringency level for inclusion (used only in simple mode, range 0 to 1).
- **model (str)**: Model to use for processing (default is "gpt-4o-mini-2024-07-18").
- **zeroshot (bool)**: If True, uses a zeroshot approach; otherwise, uses simple mode.

### Returns
- **df_results (DataFrame)**: A DataFrame containing the results of the abstract filtering, including judgement, title, abstract, and various probability scores.

## Example Usage

results = run('abstracts.csv', 'Include studies that focus on XYZ', thread=8)


This example processes abstracts from `abstracts.csv` with the specified inclusion criteria using 8 threads.

## Notes
- Ensure that the OpenAI API key is stored in the specified `api_file`.
- Adjust the `http_proxy` and `https_proxy` parameters as needed for your network configuration.
- The `k` parameter allows you to control the stringency of the inclusion criteria, where 0 is the least stringent and 1 is the most stringent.
