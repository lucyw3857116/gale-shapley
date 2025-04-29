CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

TARGET = gale_shapley_serial
SRC = galeshapley_serial.cpp galeshapley.h

all: $(TARGET)

$(TARGET): galeshapley.cpp galeshapley.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) galeshapley.cpp

clean:
	rm -f $(TARGET)
