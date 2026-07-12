use crate::math::{BBox, Vec3};
use crate::geometry::{Intersectable, HitRecord};

// --- BVH NODE ---
pub struct BVHNode {
    bbox: BBox,
    left: Box<dyn Intersectable>,
    right: Box<dyn Intersectable>,
}

impl BVHNode {
    pub fn build(mut objects: Vec<Box<dyn Intersectable>>) -> Option<Box<dyn Intersectable>> {
        if objects.is_empty() {
            return None;
        }
        if objects.len() == 1 {
            return Some(objects.pop().unwrap());
        }

        let mut total_bbox = BBox::empty();
        for obj in &objects {
            total_bbox = total_bbox.surround(obj.bounding_box());
        }

        // SAH-like split: choose axis with largest extent
        let dx = total_bbox.max.x - total_bbox.min.x;
        let dy = total_bbox.max.y - total_bbox.min.y;
        let dz = total_bbox.max.z - total_bbox.min.z;

        if dx > dy && dx > dz {
            objects.sort_by(|a, b| a.bounding_box().min.x.partial_cmp(&b.bounding_box().min.x).unwrap());
        } else if dy > dz {
            objects.sort_by(|a, b| a.bounding_box().min.y.partial_cmp(&b.bounding_box().min.y).unwrap());
        } else {
            objects.sort_by(|a, b| a.bounding_box().min.z.partial_cmp(&b.bounding_box().min.z).unwrap());
        }

        let mid = objects.len() / 2;
        let right_objs = objects.split_off(mid);
        
        let left = Self::build(objects)?;
        let right = Self::build(right_objs)?;

        Some(Box::new(BVHNode {
            bbox: total_bbox,
            left,
            right,
        }))
    }
}

impl Intersectable for BVHNode {
    fn intersect(&self, ray: &crate::geometry::Ray) -> Option<HitRecord> {
        if !self.bbox.intersect(ray) {
            return None;
        }

        let left_hit = self.left.intersect(ray);

        // Early exit: compute the minimum possible t for the right subtree.
        // If the left hit is closer than that, we can skip the right subtree entirely.
        if let Some(ref l) = left_hit {
            let min_t_right = ray.min_t_intersect(&self.right.bounding_box());
            if let Some(min_tr) = min_t_right {
                if l.t < min_tr {
                    return left_hit;
                }
            }
        }

        let right_hit = self.right.intersect(ray);

        match (left_hit, right_hit) {
            (Some(l), Some(r)) => if l.t < r.t { Some(l) } else { Some(r) },
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }

    fn bounding_box(&self) -> BBox {
        self.bbox
    }
}
