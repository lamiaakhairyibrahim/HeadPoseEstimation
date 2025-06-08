from src.collect import collect_out
import argparse

def data_org(data_org_path , video_path_):
    data_input = collect_out(path_of_data = data_org_path , video_path= video_path_)
    
def main():
    parser = argparse.ArgumentParser(description="enter the dataset path.")
    parser.add_argument('data_org_path', help="The path to incloude")
    parser.add_argument('video_path_', help="The path to incloude")
    args = parser.parse_args()

    try:
        data_org(args.data_org_path, args.video_path_)
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()