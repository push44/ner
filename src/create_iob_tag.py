import config
import pandas as pd
from tqdm import tqdm

#return file text in a tokenized list using python .split()
def return_text_list(file_id):
    with open(f"{config.train_essay_folder}/{file_id}.txt", "r") as f:
        text_list = f.read().split()
    return text_list

def create_iob_df(train_annot_df):
    #convert predictionstring to list
    train_annot_df["predictionlist"] = train_annot_df["predictionstring"].apply(lambda string: list(map(int, string.split())))
    #get unique file ids
    file_ids = train_annot_df["id"].unique().tolist()

    #create iob tag dataframe (one row for one word of a file)
    iob_dfs = []
    for file_id in tqdm(file_ids):
        #get dataframe for file_id
        sample_df = train_annot_df[train_annot_df["id"]==file_id]
        #create iob_df for file_id
        sample_iob_df = pd.DataFrame({
            "id":file_id,
            "word":return_text_list(file_id),
            "tag":"O"
        })
        #for every row in train_annot_df (multiple rows for one file_id)
        for i in range(sample_df.shape[0]):
            #get info for i'th row of a file_id
            prediction_list = sample_df["predictionlist"].iloc[i]
            discourse_type = sample_df["discourse_type"].iloc[i].lower()
            
            #starting of discourse tag is marked with B (begining)
            sample_iob_df["tag"].iloc[prediction_list[0]] = f"B-{discourse_type}"
            if len(prediction_list)>1:
                #rest of discourse tag is marked with I (inside)
                sample_iob_df["tag"].iloc[prediction_list[1]:prediction_list[-1]+1] = f"I-{discourse_type}"
        
        #append all sample_iob_df representing each file_id
        iob_dfs.append(sample_iob_df)
        
    #return single iob_df
    return pd.concat(iob_dfs).reset_index(drop=True)

if __name__ == "__main__":
    iob_df = create_iob_df(
        train_annot_df = pd.read_csv(config.TRAIN_ANNOT_FILE)
    )
    iob_df.to_csv(config.TRAIN_IOB_FILE, index=False)