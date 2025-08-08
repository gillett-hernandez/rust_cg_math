use typenum::Unsigned;

/// Euclidean space - R, R^2, R^3, etc
pub struct R<T: Unsigned>;

/// space of angles - loops on itself
pub struct Circle;

/// includes the interior
pub struct Disk;

/// only the surface - space of directions, solid angle, and surface area
pub struct Sphere;

/// includes interior
pub struct Ball;
