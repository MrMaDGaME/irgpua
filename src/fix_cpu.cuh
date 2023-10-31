#pragma once

#include "image.hh"

void fix_image_cpu(Image& to_fix);

void save_array(int *array, int size, std::string name);
