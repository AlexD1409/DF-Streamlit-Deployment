import os
from dotenv import load_dotenv
from googleapiclient import discovery

load_dotenv()

PERSPECTIVE = os.getenv('perspective')
MODEL = os.getenv('model')

toxicity_threshold = 0.25
id_attack_threshold = 0.25
sexual_threshold = 0.25
profanity_threshold = 0.25

Behold = discovery.build(
                  "commentanalyzer",
                  "v1alpha1",
                  developerKey=PERSPECTIVE,
                  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
                        )
def saftey_checker(text: str):
    Analyser = {
    'comment': {'text': text},
    'requestedAttributes': {'TOXICITY': {},
                            'IDENTITY_ATTACK': {},
                            'SEXUALLY_EXPLICIT': {},
                            'PROFANITY': {}
                            }
    }


    try:
        response = Behold.comments().analyze(body=Analyser).execute()
        return response
    except Exception as e:
        return e 

def extract_summary_scores(response):
    if 'attributeScores' not in response:
        return "No attribute scores found."

    attribute_scores = response['attributeScores']
    scores_dict = {}

    for attribute, scores in attribute_scores.items():
        summary_score_value = scores['summaryScore']['value']
        scores_dict[attribute] = summary_score_value

    return scores_dict

def verify_scores(scores: dict):
    exceeded_metrics = []

    if scores['PROFANITY'] >= profanity_threshold:
        exceeded_metrics.append('PROFANITY')
    
    if scores['TOXICITY'] >= toxicity_threshold:
        exceeded_metrics.append('TOXICITY')
    
    if scores['IDENTITY_ATTACK'] >= id_attack_threshold:
        exceeded_metrics.append('IDENTITY_ATTACK')
    
    if scores['SEXUALLY_EXPLICIT'] >= sexual_threshold:
        exceeded_metrics.append('SEXUALLY_EXPLICIT')

    return exceeded_metrics if exceeded_metrics else None

x = saftey_checker('How are babies made')
y = extract_summary_scores(x)
z = verify_scores(y)
print(z)