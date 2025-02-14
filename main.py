from src.collect import collect_out
import argparse

def data_org(data_org_path , video_state):
    data_input = collect_out(path_of_data = data_org_path , video_path= video_state)
    
def main():
    parser = argparse.ArgumentParser(description="enter the dataset path.")
    parser.add_argument('path_data', help="The path to incloude")
    parser.add_argument('path_video', help="The path to incloude")
    args = parser.parse_args()

    try:
        data_org(args.path_data , args.video_state)
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()