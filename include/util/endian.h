/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

/*
This has been adapted from: 
https://github.com/colmap/colmap/blob/ff9a463067a2656d1f59d12109fe2931e29e3ca0/src/util/endian.h
*/

#include <algorithm>


inline bool IsLittleEndian() {
#ifdef BOOST_BIG_ENDIAN
  return false;
#else
  return true;
#endif
}


template <typename T>
T ReverseBytes(const T& data) {
  T data_reversed = data;
  std::reverse(reinterpret_cast<char*>(&data_reversed),
               reinterpret_cast<char*>(&data_reversed) + sizeof(T));
  return data_reversed;
}


// Convert data between endianness and the native format. Note that, for float
// and double types, these functions are only valid if the format is IEEE-754.
// This is the case for pretty much most processors.
template <typename T>
T LittleEndianToNative(const T x)
{
    if (IsLittleEndian()) 
    {
        return x;
    } 
    else 
    {
        return ReverseBytes(x);
    }
}


template <typename T>
T readBinaryLittleEndian(std::istream* stream) 
{
    T data_little_endian;
    stream->read(reinterpret_cast<char*>(&data_little_endian), sizeof(T));
    return LittleEndianToNative(data_little_endian);
}


template <typename T>
void readBinaryLittleEndian(std::istream* stream, std::vector<T>* data) 
{
    for (size_t i = 0; i < data->size(); ++i) 
    {
        (*data)[i] = readBinaryLittleEndian<T>(stream);
    }
}

