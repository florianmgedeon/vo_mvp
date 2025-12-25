CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   = $(shell pkg-config --libs opencv4)

TARGET = vo_mvp
SRC = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $^ -o $@ $(OPENCV_LIBS)

clean:
	rm -f $(TARGET)
