from dotenv import load_dotenv

load_dotenv()

from scripts import peft_train_main

if __name__ == "__main__":
    peft_train_main()
