from preprocess import DataLoader, DataPreprocessor, DataSplitter

# File path
filepath = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Load data
data_loader = DataLoader(filepath)
df = data_loader.load_data()
df = data_loader.check_data_quality()

# Preprocess data
preprocessor = DataPreprocessor(df)
df_internet, df_phone = preprocessor.split_by_service()

# Preprocess categorical columns
df_internet = preprocessor.preprocess_categorical(df_internet)
df_phone = preprocessor.preprocess_categorical(df_phone)

# Format column names
df_internet.columns = df_internet.columns.str.lower().str.replace(' ', '_')
df_phone.columns = df_phone.columns.str.lower().str.replace(' ', '_')

# Split and process data
splitter = DataSplitter(df_internet, df_phone, 0.2)
splitter.train_test_split()
df_train_internet_only, df_test_internet_only = splitter.get_only_service_split('Internet')

# Impute and revert to booleans
df_train_internet_only, df_test_internet_only = splitter.impute_missing_values(df_train_internet_only, df_test_internet_only)

# Export
df_train_internet_only.to_csv('preprocessed_data/df_train_internet_only.csv', index = False)
df_test_internet_only.to_csv('preprocessed_data/df_test_internet_only.csv', index = False)