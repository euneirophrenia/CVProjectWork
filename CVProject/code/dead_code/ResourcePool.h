////
//// Created by Marco DiVi on 03/03/18.
////
//
//#ifndef OPENCV_RESOURCEPOOL_H
//#define OPENCV_RESOURCEPOOL_H
//
//#include <map>
//#include <functional>
//
///**
// * A caching structure for commonly shared objects, essentially relying on a hashmap.
// * @tparam K the keys' type. Use this type to reference the objects
// * @tparam T the type of the objects.
// */
////TODO: manage memory correctly, provide a destructor and destroy instances when they're no more needed.
//template <class K, class T> class ResourcePool {
//
//    private:
//        std::map<K,T*> _map = std::map<K,T*>();
//
//public:
//
//    ResourcePool<K,T>(void);
//
//    T* getOrElse(K, std::function<T*(K)>);
//
//};
//
//
//template<class K, class T>
//inline T* ResourcePool<K, T>::getOrElse(K key, std::function<T*(K)> lambda) {
//
//    if (this->_map.count(key) == 0) {
//        this->_map[key] = lambda(key);
//    }
//
//    return this->_map[key];
//}
//
//template<class K, class T>
//ResourcePool<K, T>::ResourcePool(void) {
//    //???????
//}
//
//#endif //OPENCV_RESOURCEPOOL_H
