import pandas as pd
import os
import json

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return f"Error loading CSV file: {str(e)}"

def get_column_data_types(df):
    return df.dtypes.to_dict()

def get_basic_insights(df):
    insights = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_data_types": get_column_data_types(df)
    }
    return insights

def run_data_loader(file_paths):
    context = {
        "dataframes": {},
        "insights": {}
    }
    for file_path in file_paths:
        df_name = os.path.basename(file_path).split('.')[0]
        df = load_csv(file_path)
        if isinstance(df, str):  # Error message
            context["dataframes"][df_name] = df
        else:
            context["dataframes"][df_name] = df
            context["insights"][df_name] = get_basic_insights(df)
    return context

# Example usage
if __name__ == "__main__":
    file_paths = ["sample_data.csv"]
    context = run_data_loader(file_paths)
    print(json.dumps(context, indent=2))