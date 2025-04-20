# import numpy as np
# import pandas as pd
# import os
# import logging
# from prelogad.DeepSVDD.src.deepSVDD import DeepSVDD
# from prelogad.DeepSVDD.src.datasets.main import load_dataset
# from tqdm import tqdm 
# import yaml
# from postprocess import RAG
# from utils.evaluator import evaluate
# import torch
# import openai

# # Configure logging first
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler = logging.FileHandler('./output/runtime.log')
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# # Load configs with validation
# try:
#     with open('config.yaml', 'r') as file:
#         configs = yaml.safe_load(file)
        
#     # Validate required keys
#     required_keys = ['api_key', 'llm_name', 'log_structed_path', 'dataset_name']
#     for key in required_keys:
#         if key not in configs:
#             raise ValueError(f"Missing required config key: {key}")

#     # Configure OpenAI
#     openai.api_key = configs['api_key']
#     if configs.get('api_base'):
#         os.environ["OPENAI_API_BASE"] = configs['api_base']
        
# except Exception as e:
#     logger.error(f"Configuration error: {str(e)}")
#     exit(1)

# def train_deepsvdd(train_data_path):
#     """Train DeepSVDD model"""
#     try:
#         if not os.path.exists('./output'):
#             os.makedirs('./output')
        
#         deep_SVDD = DeepSVDD('soft-boundary')
#         deep_SVDD.set_network("mlp")
        
#         if not configs.get('is_train', True):
#             deep_SVDD.load_model(model_path='./output/model.tar', load_ae=False)
#             logger.info('Loading model from ./output/model.tar')
#             return './output/model.tar'
            
#         train_dataset = load_dataset(
#             data_path=train_data_path,
#             encoder_path=configs.get('encoder_path', None)
#         )

#         if configs.get('is_pretrain', False):
#             deep_SVDD.pretrain(
#                 train_dataset,
#                 optimizer_name=configs['optimizer_name'],
#                 lr=configs['lr'],
#                 n_epochs=configs['n_epochs'],
#                 lr_milestones=configs['lr_milestones'],
#                 batch_size=configs['batch_size'],
#                 weight_decay=configs['weight_decay'],
#                 device=configs['device'],
#                 n_jobs_dataloader=configs['n_jobs_dataloader'])
                
#         deep_SVDD.train(
#             train_dataset,
#             optimizer_name=configs['optimizer_name'],
#             lr=configs['lr'],
#             n_epochs=configs['n_epochs'],
#             lr_milestones=configs['lr_milestones'],
#             batch_size=configs['batch_size'],
#             weight_decay=configs['weight_decay'],
#             device=configs['device'],
#             n_jobs_dataloader=configs['n_jobs_dataloader'])
            
#         model_path = './output/model.tar'
#         deep_SVDD.save_results(export_json='./output/results.json')
#         deep_SVDD.save_model(export_model=model_path, save_ae=False)
#         return model_path
        
#     except Exception as e:
#         logger.error(f"Training failed: {str(e)}")
#         raise

# def anomaly_detection(model_path, test_data_path):
#     """Run anomaly detection"""
#     try:
#         logger.info("Starting testing...")
        
#         deep_SVDD = DeepSVDD('soft-boundary')
#         deep_SVDD.set_network("mlp")
#         deep_SVDD.load_model(model_path=model_path, load_ae=False)
        
#         test_dataset = load_dataset(
#             data_path=test_data_path,
#             encoder_path=configs.get('encoder_path', None))
            
#         anomalies, _ = deep_SVDD.test(
#             test_dataset,
#             device=configs['device'],
#             n_jobs_dataloader=configs['n_jobs_dataloader'])
           
#         anomaly_lineid_list = [item[0] for item in tqdm(anomalies, desc='Processing anomalies')]
        
#         # Save results
#         output_file = 'output/anomaly_logs_detc_by_svdd.csv'
#         df_test = pd.read_csv(test_data_path)
#         pos_df = df_test[df_test["LineId"].isin(anomaly_lineid_list)]
#         pos_df.to_csv(output_file, index=False)
        
#         return output_file, anomaly_lineid_list
        
#     except Exception as e:
#         logger.error(f"Anomaly detection failed: {str(e)}")
#         raise

# def main():
#     try:
#         logger.info("Starting LogRAG with config: %s", configs)
        
#         # Prepare data
#         all_df = pd.read_csv(configs['log_structed_path'])
#         num_train = int(configs['train_ratio']*len(all_df))
#         train_df = all_df[:num_train]
#         test_df = all_df[num_train:]
        
#         # Ensure output directory exists
#         os.makedirs('./output', exist_ok=True)
#         os.makedirs(f"./dataset/{configs['dataset_name']}", exist_ok=True)
        
#         # Save split data
#         train_path = f"./dataset/{configs['dataset_name']}/train_log_structured.csv"
#         test_path = f"./dataset/{configs['dataset_name']}/test_log_structured.csv"
#         train_df.to_csv(train_path, index=False)
#         test_df.to_csv(test_path, index=False)
        
#         # Training and detection
#         model_path = train_deepsvdd(train_path)
#         anomaly_logs_path, anomaly_lineid_list = anomaly_detection(model_path, test_path)
        
#         # RAG processing if enabled
#         if configs.get('is_rag', False):
#             logger.info("Starting RAG post-processing")
#             RagPoster = RAG.RAGPostProcessor(
#                 configs,
#                 train_data_path=train_path,
#                 logger=logger)
#             anomaly_lineid_list = RagPoster.post_process(
#                 anomaly_logs_path,
#                 test_path)
                
#         # Evaluation
#         evaluate(configs, test_path, anomaly_lineid_list, logger)
        
#     except Exception as e:
#         logger.error(f"Main execution failed: {str(e)}")
#         raise

# if __name__ == '__main__':
#     main()




















import numpy as np
import pandas as pd
import os
import logging
from prelogad.DeepSVDD.src.deepSVDD import DeepSVDD
from prelogad.DeepSVDD.src.datasets.main import load_dataset
from tqdm import tqdm 
import yaml
from postprocess import RAG
from utils.evaluator import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('./output/runtime.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load configs with validation
try:
    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
        
    # Validate required keys for Mistral
    required_keys = ['llm_name', 'log_structed_path', 'dataset_name', 'encoder_path']
    for key in required_keys:
        if key not in configs:
            raise ValueError(f"Missing required config key: {key}")

    # Verify Mistral model path exists
    if not os.path.exists(configs['encoder_path']):
        raise FileNotFoundError(f"Mistral model not found at {configs['encoder_path']}")
        
except Exception as e:
    logger.error(f"Configuration error: {str(e)}")
    exit(1)

def load_mistral_model():
    """Load Mistral model with quantization"""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        configs['encoder_path'],
        quantization_config=quant_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        configs['encoder_path'],
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def train_deepsvdd(train_data_path):
    """Train DeepSVDD model"""
    try:
        os.makedirs('./output', exist_ok=True)
        
        deep_SVDD = DeepSVDD('soft-boundary')
        deep_SVDD.set_network("mlp")
        
        if not configs.get('is_train', True):
            deep_SVDD.load_model(model_path='./output/model.tar', load_ae=False)
            logger.info('Loading model from ./output/model.tar')
            return './output/model.tar'
            
        # Load dataset with Mistral tokenizer
        train_dataset = load_dataset(
            data_path=train_data_path,
            encoder_path=configs['encoder_path']
        )

        if configs.get('is_pretrain', False):
            deep_SVDD.pretrain(
                train_dataset,
                optimizer_name=configs['optimizer_name'],
                lr=configs['lr'],
                n_epochs=configs['n_epochs'],
                lr_milestones=configs['lr_milestones'],
                batch_size=configs['batch_size'],
                weight_decay=configs['weight_decay'],
                device=configs['device'],
                n_jobs_dataloader=configs['n_jobs_dataloader'])
                
        deep_SVDD.train(
            train_dataset,
            optimizer_name=configs['optimizer_name'],
            lr=configs['lr'],
            n_epochs=configs['n_epochs'],
            lr_milestones=configs['lr_milestones'],
            batch_size=configs['batch_size'],
            weight_decay=configs['weight_decay'],
            device=configs['device'],
            n_jobs_dataloader=configs['n_jobs_dataloader'])
            
        model_path = './output/model.tar'
        deep_SVDD.save_results(export_json='./output/results.json')
        deep_SVDD.save_model(export_model=model_path, save_ae=False)
        return model_path
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def anomaly_detection(model_path, test_data_path):
    """Run anomaly detection"""
    try:
        logger.info("Starting testing...")
        
        deep_SVDD = DeepSVDD('soft-boundary')
        deep_SVDD.set_network("mlp")
        deep_SVDD.load_model(model_path=model_path, load_ae=False)
        
        test_dataset = load_dataset(
            data_path=test_data_path,
            encoder_path=configs['encoder_path'])
            
        anomalies, _ = deep_SVDD.test(
            test_dataset,
            device=configs['device'],
            n_jobs_dataloader=configs['n_jobs_dataloader'])
           
        anomaly_lineid_list = [item[0] for item in tqdm(anomalies, desc='Processing anomalies')]
        
        output_file = 'output/anomaly_logs_detc_by_svdd.csv'
        df_test = pd.read_csv(test_data_path)
        pos_df = df_test[df_test["LineId"].isin(anomaly_lineid_list)]
        pos_df.to_csv(output_file, index=False)
        
        return output_file, anomaly_lineid_list
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting LogRAG-Mistral with config: %s", configs)
        
        # Load Mistral model early to verify it works
        if "mistral" in configs['llm_name'].lower():
            model, tokenizer = load_mistral_model()
            logger.info("Successfully loaded Mistral model")
        
        # Prepare data
        all_df = pd.read_csv(configs['log_structed_path'])
        num_train = int(configs['train_ratio']*len(all_df))
        train_df = all_df[:num_train]
        test_df = all_df[num_train:]
        
        # Save split data
        os.makedirs(f"./dataset/{configs['dataset_name']}", exist_ok=True)
        train_path = f"./dataset/{configs['dataset_name']}/train_log_structured.csv"
        test_path = f"./dataset/{configs['dataset_name']}/test_log_structured.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Training and detection
        model_path = train_deepsvdd(train_path)
        anomaly_logs_path, anomaly_lineid_list = anomaly_detection(model_path, test_path)
        
        # RAG processing with Mistral
        if configs.get('is_rag', False):
            logger.info("Starting Mistral RAG post-processing")
            RagPoster = RAG.RAGPostProcessor(
                configs,
                train_data_path=train_path,
                logger=logger,
                llm_model=model,
                llm_tokenizer=tokenizer
            )
            anomaly_lineid_list = RagPoster.post_process(
                anomaly_logs_path,
                test_path)
                
        # Evaluation
        evaluate(configs, test_path, anomaly_lineid_list, logger)
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()