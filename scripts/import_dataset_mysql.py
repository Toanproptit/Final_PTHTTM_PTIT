"""Import the validated Kaggle CSV into the configured relational database."""
from diabetes_demo.dataset_repository import import_dataset
from diabetes_demo.storage import DemoStore

if __name__ == "__main__":
    store = DemoStore(create_schema=False)
    print(import_dataset(store))
