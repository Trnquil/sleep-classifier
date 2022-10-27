from source.preprocessing.collection import Collection

class CollectionService(object):
    @staticmethod
    def crop(motion_collection, interval):
        subject_id = motion_collection.subject_id
        timestamps = motion_collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]

        cropped_data = motion_collection.data[valid_indices, :]
        return Collection(subject_id=subject_id, data=cropped_data)