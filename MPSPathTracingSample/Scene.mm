/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 Implementation for scene creation functions
 */


#import "Scene.h"

using namespace simd;

std::vector<vector_float3> vertices;
std::vector<vector_float3> normals;
std::vector<vector_float2> textureCoords;
std::vector<uint32_t> textureIndices;
std::vector<float> reflections;
std::vector<float> refractions;
std::vector<vector_float3> colors;
std::vector<uint32_t> masks;

float3 getTriangleNormal(float3 v0, float3 v1, float3 v2) {
    float3 e1 = normalize(v1 - v0);
    float3 e2 = normalize(v2 - v0);
    
    return cross(e1, e2);
}

void createCubeFace(std::vector<float3> & vertices,
                    std::vector<float3> & normals,
                    std::vector<float2> & textureCoords,
                    std::vector<float3> & colors,
                    float3 *cubeVertices,
                    float3 color,
                    unsigned int i0,
                    unsigned int i1,
                    unsigned int i2,
                    unsigned int i3,
                    bool inwardNormals,
                    unsigned int textureIndex,
                    float reflection,
                    float refraction,
                    unsigned int triangleMask)
{
    float3 v0 = cubeVertices[i0];
    float3 v1 = cubeVertices[i1];
    float3 v2 = cubeVertices[i2];
    float3 v3 = cubeVertices[i3];
    
    float3 n0 = getTriangleNormal(v0, v1, v2);
    float3 n1 = getTriangleNormal(v0, v2, v3);
    
    if (inwardNormals) {
        n0 = -n0;
        n1 = -n1;
    }
    
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    vertices.push_back(v0);
    vertices.push_back(v2);
    vertices.push_back(v3);

    for (int i = 0; i < 6; i++) {
        reflections.push_back(reflection);
        refractions.push_back(refraction);
        textureIndices.push_back(textureIndex);
    }

    textureCoords.push_back(vector2(0.0f, 0.0f));
    textureCoords.push_back(vector2(0.0f, 1.0f));
    textureCoords.push_back(vector2(1.0f, 1.0f));
    
    textureCoords.push_back(vector2(0.0f, 0.0f));
    textureCoords.push_back(vector2(1.0f, 1.0f));
    textureCoords.push_back(vector2(1.0f, 0.0f));

    for (int i = 0; i < 3; i++)
        normals.push_back(n0);
    
    for (int i = 0; i < 3; i++)
        normals.push_back(n1);
    
    for (int i = 0; i < 6; i++)
        colors.push_back(color);
    
    for (int i = 0; i < 2; i++)
        masks.push_back(triangleMask);
}

void createCube(unsigned int faceMask,
                vector_float3 color,
                matrix_float4x4 transform,
                bool inwardNormals,
                unsigned int textureIndex,
                float reflection,
                float refraction,
                unsigned int triangleMask)
{
    float3 cubeVertices[] = {
        vector3(-0.5f, -0.5f, -0.5f),
        vector3( 0.5f, -0.5f, -0.5f),
        vector3(-0.5f,  0.5f, -0.5f),
        vector3( 0.5f,  0.5f, -0.5f),
        vector3(-0.5f, -0.5f,  0.5f),
        vector3( 0.5f, -0.5f,  0.5f),
        vector3(-0.5f,  0.5f,  0.5f),
        vector3( 0.5f,  0.5f,  0.5f),
    };
    
    for (int i = 0; i < 8; i++) {
        float3 vertex = cubeVertices[i];
        
        float4 transformedVertex = vector4(vertex.x, vertex.y, vertex.z, 1.0f);
        transformedVertex = transform * transformedVertex;
        
        cubeVertices[i] = transformedVertex.xyz;
    }
    
    if (faceMask & FACE_MASK_NEGATIVE_X)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 0, 4, 6, 2, inwardNormals, textureIndex, reflection, refraction, triangleMask);
    
    if (faceMask & FACE_MASK_POSITIVE_X)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 1, 3, 7, 5, inwardNormals, textureIndex, reflection, refraction, triangleMask);
    
    if (faceMask & FACE_MASK_NEGATIVE_Y)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 0, 1, 5, 4, inwardNormals, textureIndex, reflection, refraction, triangleMask);
    
    if (faceMask & FACE_MASK_POSITIVE_Y)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 2, 6, 7, 3, inwardNormals, textureIndex, reflection, refraction, triangleMask);
    
    if (faceMask & FACE_MASK_NEGATIVE_Z)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 0, 2, 3, 1, inwardNormals, textureIndex, reflection, refraction, triangleMask);
    
    if (faceMask & FACE_MASK_POSITIVE_Z)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 4, 5, 7, 6, inwardNormals, textureIndex, reflection, refraction, triangleMask);
}

void createTriangle(std::vector<float3> & vertices,
                    std::vector<float3> & normals,
                    std::vector<float2> & textureCoords,
                    std::vector<float3> & colors,
                    float3 *triangleVertices,
                    float3 *triangleNormal,
                    float2 *triangletextureCoords,
                    float3 color,
                    unsigned int textureIndex,
                    float reflection,
                    float refraction,
                    unsigned int triangleMask)
{
    for (int i = 0; i < 3; i++) {
        reflections.push_back(reflection);
        refractions.push_back(refraction);
        textureIndices.push_back(textureIndex);
    }
    
    for (int i = 0; i < 3; i++) {
        vertices.push_back(triangleVertices[i]);
        normals.push_back(triangleNormal[i]);
        textureCoords.push_back(triangletextureCoords[i]);
        colors.push_back(color);
    }
    
    masks.push_back(triangleMask);
}

#define RING_STEPS 30
#define POINT_STEPS 30

void createSphere(vector_float3 color,
                  matrix_float4x4 transform,
                  unsigned int textureIndex,
                  float reflection,
                  float refraction,
                  unsigned int triangleMask) {
    float pi = 22.0f / 7.0f;
    float deltaTheta = pi / (RING_STEPS + 2);
    float deltaPhi = 2.0f * pi / POINT_STEPS;
    float theta = 0.0f;
    
    for (int ring = 0; ring < RING_STEPS + 2; ring++) {
        float phi = 0.0f;
        for (int point = 0; point < POINT_STEPS; point++) {
            float3 v0 = vector3(sin(theta) * cos(phi),
                               sin(theta) * sin(phi),
                               cos(theta));

            float3 v1 = vector3(sin(theta + deltaTheta) * cos(phi),
                               sin(theta + deltaTheta) * sin(phi),
                               cos(theta + deltaTheta));

            float3 v2 = vector3(sin(theta) * cos(phi + deltaPhi),
                               sin(theta) * sin(phi + deltaPhi),
                                cos(theta));
            
            float3 v3 = vector3(sin(theta + deltaTheta) * cos(phi + deltaPhi),
                               sin(theta + deltaTheta) * sin(phi + deltaPhi),
                                cos(theta + deltaTheta));
            
            float3 triangleVertices0[] = {
                (transform * vector4(v0, 1.0f)).xyz,
                (transform * vector4(v1, 1.0f)).xyz,
                (transform * vector4(v2, 1.0f)).xyz };

            float3 normal0[] = {
                (transform * vector4(v0, 0.0f)).xyz,
                (transform * vector4(v1, 0.0f)).xyz,
                (transform * vector4(v2, 0.0f)).xyz};
            
            float2 textureCoords0[] = {0.0f, 1.0f};
            createTriangle(vertices, normals, textureCoords, colors, triangleVertices0, normal0, textureCoords0, color, textureIndex, reflection, refraction, triangleMask);
            
            float3 triangleVertices1[] = {
                (transform * vector4(v3, 1.0f)).xyz,
                (transform * vector4(v2, 1.0f)).xyz,
                (transform * vector4(v1, 1.0f)).xyz};

            float3 normal1[] = {
                (transform * vector4(v3, 0.0f)).xyz,
                (transform * vector4(v2, 0.0f)).xyz,
                (transform * vector4(v1, 0.0f)).xyz};
            
            float2 textureCoords1[] = {0.0f, 1.0f};
            createTriangle(vertices, normals, textureCoords, colors, triangleVertices1, normal1, textureCoords1, color, textureIndex, reflection, refraction, triangleMask);
            phi += deltaPhi;
        }
        theta += deltaTheta;
    }
    
//    drawVertex(0,0,1) //north pole end cap
//    drawVertex(0, 0, -1) //south pole end cap
}

void createMesh(MDLMesh* mesh, vector_float3 color, matrix_float4x4 transform, unsigned int textureIndex, float reflection, float refraction, unsigned int triangleMask) {
    for (int i = 0; i < (int)mesh.submeshes.count; i++) {
        MDLSubmesh *subMesh = mesh.submeshes[i];
        NSLog(@"Submesh Index count: %d", (int)subMesh.indexCount);

        MDLMeshBufferData *meshBufferForIndices = subMesh.indexBuffer;
        NSLog(@"Meshbuffer Index count: %d", (int)subMesh.indexCount);
        uint32_t *indices = (uint32_t *)meshBufferForIndices.data.bytes;
        
        NSArray <id<MDLMeshBuffer>> *arrayOfMeshBuffers = mesh.vertexBuffers;
        MDLMeshBufferData *meshBufferForVertice = arrayOfMeshBuffers[0];
        NSLog(@"Vertex Buffer Count: %d", (int)meshBufferForVertice.length / 4);
        float *vertexData = (float *)meshBufferForVertice.data.bytes;

        NSLog(@"Vertex Buffer Count: %d", (int)mesh.vertexCount);
        for (int i = 0; i < (int)subMesh.indexCount; i+=3) {
//            NSLog(@"Triangle Indices: %d, %d, %d", (int)indices[i], (int)indices[i+1], (int)indices[i+2]);
            float3 v0 = vector3((float)vertexData[indices[i+0]*8 + 0], (float)vertexData[indices[i+0]*8 + 1], (float)vertexData[indices[i+0]*8 + 2]);
            float3 v1 = vector3((float)vertexData[indices[i+1]*8 + 0], (float)vertexData[indices[i+1]*8 + 1], (float)vertexData[indices[i+1]*8 + 2]);
            float3 v2 = vector3((float)vertexData[indices[i+2]*8 + 0], (float)vertexData[indices[i+2]*8 + 1], (float)vertexData[indices[i+2]*8 + 2]);
            float3 triangleVertices[] = {
                (transform * vector4(v0, 1.0f)).xyz,
                (transform * vector4(v1, 1.0f)).xyz,
                (transform * vector4(v2, 1.0f)).xyz };

            float3 n0 = vector3((float)vertexData[indices[i+0]*8 + 3], (float)vertexData[indices[i+0]*8 + 4], (float)vertexData[indices[i+0]*8 + 5]);
            float3 n1 = vector3((float)vertexData[indices[i+1]*8 + 3], (float)vertexData[indices[i+1]*8 + 4], (float)vertexData[indices[i+1]*8 + 5]);
            float3 n2 = vector3((float)vertexData[indices[i+2]*8 + 3], (float)vertexData[indices[i+2]*8 + 4], (float)vertexData[indices[i+2]*8 + 5]);
            float3 triangleNormals[] = {
                (transform * vector4(n0, 0.0f)).xyz,
                (transform * vector4(n1, 0.0f)).xyz,
                (transform * vector4(n2, 0.0f)).xyz };

            float2 t0 = vector2((float)vertexData[indices[i+0]*8 + 6], 1.0f - (float)vertexData[indices[i+0]*8 + 7]);
            float2 t1 = vector2((float)vertexData[indices[i+1]*8 + 6], 1.0f - (float)vertexData[indices[i+1]*8 + 7]);
            float2 t2 = vector2((float)vertexData[indices[i+2]*8 + 6], 1.0f - (float)vertexData[indices[i+2]*8 + 7]);
            float2 triangleTextureCoords[] = { t0, t1, t2};

            createTriangle(vertices, normals, textureCoords, colors, triangleVertices, triangleNormals, triangleTextureCoords, color, textureIndex, reflection, refraction, triangleMask);
            
//            NSLog(@"Vertex Position: %f, %f, %f", (float)vertexData[i*8 + 0], (float)vertexData[i*8 + 1], (float)vertexData[i*8 + 2]);
//            NSLog(@"Vertex Normal: %f, %f, %f", (float)vertexData[i*8 + 3], (float)vertexData[i*8 + 4], (float)vertexData[i*8 + 5]);
//            NSLog(@"Vertex Texture Coord: %f, %f", (float)vertexData[i*8 + 6], (float)vertexData[i*8 + 7]);
        }
    }
}
