/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 Header for scene creation functions
 */

#ifndef Scene_h
#define Scene_h

#include <vector>
#include <simd/simd.h>
#import <ModelIO/ModelIO.h>

extern std::vector<vector_float3> vertices;
extern std::vector<vector_float3> normals;
extern std::vector<vector_float2> textureCoords;
extern std::vector<uint32_t> textureIndices;
extern std::vector<float> reflections;
extern std::vector<float> reflectionBlurs;
extern std::vector<float> refractions;
extern std::vector<float> refractionIndices;
extern std::vector<vector_float3> colors;
extern std::vector<uint32_t> masks;

#define FACE_MASK_NONE       0
#define FACE_MASK_NEGATIVE_X (1 << 0)
#define FACE_MASK_POSITIVE_X (1 << 1)
#define FACE_MASK_NEGATIVE_Y (1 << 2)
#define FACE_MASK_POSITIVE_Y (1 << 3)
#define FACE_MASK_NEGATIVE_Z (1 << 4)
#define FACE_MASK_POSITIVE_Z (1 << 5)
#define FACE_MASK_ALL        ((1 << 6) - 1)

void createCube(unsigned int faceMask,
                vector_float3 color,
                matrix_float4x4 transform,
                bool inwardNormals,
                unsigned int textureIndex,
                float reflection,
                float reflectionBlur,
                float refraction,
                float refractionIndex,
                unsigned int triangleMask);

void createSphere(vector_float3 color,
                  matrix_float4x4 transform,
                  unsigned int textureIndex,
                  float reflection,
                  float reflectionBlur,
                  float refraction,
                  float refractionIndex,
                  unsigned int triangleMask);

void createMesh(MDLMesh* mesh,
                vector_float3 color,
                matrix_float4x4 transform,
                unsigned int textureIndex,
                float reflection,
                float reflectionBlur,
                float refraction,
                float refractionIndex,
                unsigned int triangleMask);

#endif /* Scene_h */
