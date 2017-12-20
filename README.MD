# Starter Code for MemexQA dataset


# Dataset website
	https://memexqa.cs.cmu.edu/index.html

	Explore dataset: https://memexqa.cs.cmu.edu/explore/1.html

# Dataset download
	v1.1
	(You don't have to download "shown_photos.tgz" if you use our image features "photos_inception_resnet_v2_l2norm.npz")
		memexqa_dataset_v1.1/
		├── album_info.json   # album data: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/album_info.json
		├── glove.6B.100d.txt # word vectors for baselines:  http://nlp.stanford.edu/data/glove.6B.zip
		├── shown_photos.tgz (7.5GB)  # all photos: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/shown_photos.tgz
		├── photos_inception_resnet_v2_l2norm.npz # photo features: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz
		├── qas.json # QA data: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/qas.json
		└── test_question.ids # testset Id: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/test_question.ids


# Starter Code: word embedding w/ simple LSTM
	Dependencies: tensorflow >= 1.0, tqdm, numpy...

	Code Structure:
		1. main.py # main code for training and testing
		2. model.py # the LSTM model
		3. preprocess.py # code to preprocess data into pickle files for main.py
		4. tester.py # tester class
		5. trainer.py # trainer class
		6. utils.py # some helpful function for the model and evaluation

	1 preprocess

		$ python starter_code/preprocess.py memexqa_dataset_v1.1/qas.json memexqa_dataset_v1.1/album_info.json memexqa_dataset_v1.1/test_question.ids memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz memexqa_dataset_v1.1/glove.6B.100d.txt started_test/prepro

	2 train

		$ python starter_code/main.py starter_test/prepro/ starter_test/baselines --modelname simple_lstm --keep_prob 0.7 --is_train

	3 test

		$ python starter_code/main.py starter_test/prepro/ starter_test/baselines --modelname simple_lstm --use_char --keep_prob 0.7 --is_test --load_best

		Trained for 100 epoch, testing accuracy: 