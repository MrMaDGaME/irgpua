#pragma once

#include <iostream>
#include "image.hh"
#include "pipeline.hh"
#include "fix_gpu.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

void fix_image_indus(Image& to_fix);
