//
//     _____                   __  
//    / ___/__________  __  __/ /_ 
//    \__ \/ ____/ __ \/ / /   __/
//   ___/   /___/ /_/ / /_/ / /__ 
//  /____/\____/\____/\____/\___/
//
//       Visualization Team 
//     Advanced Computing Lab 
//  Los Alamos National Laboratory
// --------------------------------
// 
// $Id: Rotation.cpp,v 1.4 2005/09/02 09:49:30 groth Exp $
// $Date: 2005/09/02 09:49:30 $
// $Revision: 1.4 $
//
// Authors: A. McPherson, P. McCormick, J. Kniss, G. Roth
//
//.........................................................................
// 
//! NOTE: Everything is in radians!
//.........................................................................
//

#include "Rotation.h"  // Class definition
#include "QuatVector.h"// For member operations
//#include <gpu.h>       // for GL calls
#include <cmath>       // For trigonometry operations

Rotation::Rotation()
{
  clear();

} // Rotation()

Rotation::Rotation(const Rotation &rot)
{
  setQuaternion(rot.m_quat);

} // Rotation()
  
Rotation::Rotation(float angle, float x, float y, float z)
{
  set(angle, x, y, z);

} // Rotation()
  
Rotation::~Rotation()
{

} // ~Rotation()

//! Set rotation according to angle and axis
void Rotation::set(float angle, float x, float y, float z)
{
  m_quat.set(x, y, z);
  m_quat.normalize();
  m_quat.scale(sin(angle/2.0));
  m_quat[3] = cos(angle/2.0);

} // set()

//! Set quaternaion based on angle and axis
void Rotation::set(float angle, QuatVector &axis)
{
  m_quat.copy(axis);
  m_quat.normalize();
  m_quat.scale(sin(angle/2.0));
  m_quat[3] = cos(angle/2.0);

} // set()

//! Accumulate rotation according to angle and axis
void Rotation::rotate(float angle, float x, float y, float z)
{
  QuatVector quat(x, y, z);
  quat.normalize();
  quat.scale(sin(angle/2.0));
  quat[3] = cos(angle/2.0);

  addQuaternion(quat);

} // rotate()

//! Accumulate rotation according to angle and axis
void Rotation::rotate(float angle, QuatVector &axis)
{
  QuatVector quat(axis);
  quat.normalize();
  quat.scale(sin(angle/2.0));
  quat[3] = cos(angle/2.0);

  addQuaternion(quat);

} // rotate()

//! Set rotation according to explicit quaternion specification 
void Rotation::setQuaternion(float x, float y, float z, float w)
{
  m_quat.set(x, y, z, w);

} // setQuaternion()

void Rotation::setQuaternion(float vec[4])
{
  m_quat.set(vec);

} // setQuaternion()

void Rotation::setQuaternion(const QuatVector &quat)
{
  m_quat.copy(quat);

} // setQuaternion()

//! Accumulate rotation according to explicit quaternion specification 
void Rotation::addQuaternion(float x, float y, float z, float w)
{
  addQuaternion(QuatVector(x, y, z, w));

} // addQuaternion()

void Rotation::addQuaternion(float vec[4])
{
  addQuaternion(QuatVector(vec));

} // addQuaternion()

//! Accumulate given rotation represented by a quaternion vector with this
//! into an equivalent single rotation.
//! This routine also normalizes the result every m_renorm_count times it is
//! called, to keep error from creeping in.
//! NOTE: This is a premultiply for some cursed quaternion black magick reason.
void Rotation::addQuaternion(const QuatVector &quat)
{
  static int count=0;
  QuatVector t1, t2, t3;
  QuatVector tf;

  t1.copy(m_quat);
  t1.scale(quat[3]);

  t2.copy(quat);
  t2.scale(m_quat[3]);

  t3.cross(quat, m_quat);

  tf.add(t1, t2);
  tf.add(t3, tf);

  tf[3] = quat[3] * m_quat[3] - quat.dot(m_quat);

  m_quat.copy(tf);

  if (++count > m_renorm_count)
    {
      count = 0;
      normalize();
    }

} // addQuaternion()

//! Clear rotation by clearing internal quaternion.
void Rotation::clear()
{
  m_quat.set(0, 0, 0, 1);

} // set()

//! 
void Rotation::add(const Rotation &rot)
{
  addQuaternion(rot.m_quat);

} // add()

//! Quaternions always obey:  a^2 + b^2 + c^2 + d^2 = 1.0
//! If they don't add up to 1.0, dividing by their magnitued will
//! renormalize them.
void Rotation::normalize()
{
    float mag = (m_quat[0]*m_quat[0] + m_quat[1]*m_quat[1] +
		 m_quat[2]*m_quat[2] + m_quat[3]*m_quat[3]);
    m_quat[0] /= mag;
    m_quat[1] /= mag;
    m_quat[2] /= mag;
    m_quat[3] /= mag;
    
} // normalize()

//! Build inverse rotation matrix from the internal quaternion vector
const void Rotation::matrix(float m[16])
{
  m[0] = 1.0 - 2.0 * (m_quat[1] * m_quat[1] + m_quat[2] * m_quat[2]);
  m[1] = 2.0 * (m_quat[0] * m_quat[1] + m_quat[2] * m_quat[3]);
  m[2] = 2.0 * (m_quat[2] * m_quat[0] - m_quat[1] * m_quat[3]);
  m[3] = 0.0;

  m[4] = 2.0 * (m_quat[0] * m_quat[1] - m_quat[2] * m_quat[3]);
  m[5] = 1.0 - 2.0 * (m_quat[2] * m_quat[2] + m_quat[0] * m_quat[0]);
  m[6] = 2.0 * (m_quat[1] * m_quat[2] + m_quat[0] * m_quat[3]);
  m[7] = 0.0;

  m[8] = 2.0 * (m_quat[2] * m_quat[0] + m_quat[1] * m_quat[3]);
  m[9] = 2.0 * (m_quat[1] * m_quat[2] - m_quat[0] * m_quat[3]);
  m[10] = 1.0 - 2.0 * (m_quat[1] * m_quat[1] + m_quat[0] * m_quat[0]);
  m[11] = 0.0;

  m[12] = 0.0;
  m[13] = 0.0;
  m[14] = 0.0;
  m[15] = 1.0;

  //invMatrix(m);

} // matrix()

//! Build an equivalent rotation matrix from the internal quaternion vector
void Rotation::invMatrix(float m[16])
{
  // Lucky the inverse of a rotation matrix is just the transpose
  m[0] = 1.0 - 2.0 * (m_quat[1] * m_quat[1] + m_quat[2] * m_quat[2]);
  m[1] = 2.0 * (m_quat[0] * m_quat[1] - m_quat[2] * m_quat[3]);
  m[2] = 2.0 * (m_quat[2] * m_quat[0] + m_quat[1] * m_quat[3]);
  m[3] = 0.0;

  m[4] = 2.0 * (m_quat[0] * m_quat[1] + m_quat[2] * m_quat[3]);
  m[5] = 1.0 - 2.0 * (m_quat[2] * m_quat[2] + m_quat[0] * m_quat[0]);
  m[6] = 2.0 * (m_quat[1] * m_quat[2] - m_quat[0] * m_quat[3]);
  m[7] = 0.0;

  m[8] = 2.0 * (m_quat[2] * m_quat[0] - m_quat[1] * m_quat[3]);
  m[9] = 2.0 * (m_quat[1] * m_quat[2] + m_quat[0] * m_quat[3]);
  m[10] = 1.0 - 2.0 * (m_quat[1] * m_quat[1] + m_quat[0] * m_quat[0]);
  m[11] = 0.0;

  m[12] = 0.0;
  m[13] = 0.0;
  m[14] = 0.0;
  m[15] = 1.0;

} // invMatrix()


//! Copy out quaternion
void Rotation::quaternion(float q[4])
{
  q[0] = m_quat[0];
  q[1] = m_quat[1];
  q[2] = m_quat[2];
  q[3] = m_quat[3];

} // quaternion()

void Rotation::apply()
{
  float m[16];
  matrix(m);
//  glMultMatrixf(m);
  // Assume that we are in modelview matrix mode
  //glRotatef(angle, axis[0], axis[1], axis[2]);

} // apply()

void Rotation::iapply()
{
  float m[16];
  invMatrix(m);
//  glMultMatrixf(m);
  // Assume that we are in modelview matrix mode
  //glRotatef(angle, axis[0], axis[1], axis[2]);

} // apply()
