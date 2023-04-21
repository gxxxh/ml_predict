import pandas as pd
import os





def aggregate(dir):
    # Aggregate all the csv files in the directory
    # and return a dataframe
    for root, _, files in os.walk(dir):
        i = 0
        for file in files:
            if file.endswith(".csv") :
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                df["file"] = file_path
                if 'df_all' not in locals():
                    df_all = df
                else:
                    df_all = pd.concat([df_all, df], ignore_index=True)
            i+=1
            if i%50==0:
                print("forward kernel "+str(len(df_all[df_all["dir"]=="fprop"]["kernel"].unique())))
                print("backward kernel " + str(len(df_all[df_all["dir"] == "bprop"]["kernel"].unique())))
    return df_all


if __name__ == '__main__':
   df_all = aggregate("/root/guohao/ml_predict/out/benchmark/conv2d")
   print(df_all.describe())