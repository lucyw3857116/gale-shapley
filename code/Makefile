CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

TARGET = gale_shapley_serial
SRC = galeshapley_serial.cpp galeshapley.h

all: $(TARGET)

$(TARGET): galeshapley_serial.cpp galeshapley.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) galeshapley_serial.cpp

clean:
	rm -f $(TARGET)
