CC = g++

all:
	$(CC) src/main.cpp src/TopicModel/*.cpp -o gdmr -O3 -std=c++11 -march=native -DNDEBUG -lpthread -pthread