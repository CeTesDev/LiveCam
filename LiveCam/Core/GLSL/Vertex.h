#ifndef LIVECAM_OPENGL_VERTEX_H
#define LIVECAM_OPENGL_VERTEX_H

#include <GLSL/GL.h>

namespace gl {

  struct VertexP {
    VertexP();
    VertexP(cv::Point3f pos);

    cv::Point3f pos;
  };

  struct VertexCP {
    VertexCP();
    VertexCP(cv::Vec4f color, cv::Point3f pos);

    cv::Vec4f color;
    cv::Point3f pos;
  };

  struct VertexPT {
    VertexPT();
    VertexPT(cv::Point2f pos, cv::Vec2f tex);

    cv::Point2f pos;
    cv::Vec2f tex;
  };

  struct VertexNP {
    VertexNP();
    VertexNP(cv::Point3f norm, cv::Point3f pos);

   cv::Point3f norm;
   cv::Point3f pos;
 
  };

  struct VertexNPT {
    VertexNPT();
    VertexNPT(cv::Point3f pos, cv::Point3f norm, cv::Vec2f tex);

    cv::Point3f norm;
    cv::Point3f pos;
    cv::Vec2f tex;
  };

  // ----------------------------------

  inline VertexP::VertexP() {
  }
  
  inline VertexP::VertexP(cv::Point3f pos)
    :pos(pos) 
  {
  }

  inline VertexCP::VertexCP() {
  }
  
  inline VertexCP::VertexCP(cv::Vec4f color, cv::Point3f pos)
    :color(color)
    ,pos(pos)
  {
  }

  inline VertexPT::VertexPT() {
  }

  inline VertexPT::VertexPT(cv::Point2f pos, cv::Vec2f tex)
    :pos(pos)
    ,tex(tex) 
  {
  }

  inline VertexNP::VertexNP() {
  }

  inline VertexNP::VertexNP(cv::Point3f norm, cv::Point3f pos)
    :norm(norm) 
    ,pos(pos)
  {
  }

  inline VertexNPT::VertexNPT() {
  }
  
  inline VertexNPT::VertexNPT(cv::Point3f norm, cv::Point3f pos, cv::Vec2f tex)
    :pos(pos)
    ,norm(norm)
    ,tex(tex)
  {
  }

} // gl

#endif
