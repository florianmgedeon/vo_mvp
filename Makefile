CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   = $(shell pkg-config --libs opencv4)

TARGET = vo_mvp
SRC = src/main.cpp
HDR = src/config.hpp

.PHONY: all clean re

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(SRC) -o $@ $(OPENCV_LIBS)

clean:
	rm -f $(TARGET)
	rm -rf data/frames
	rm -rf out/matches

re: clean all
