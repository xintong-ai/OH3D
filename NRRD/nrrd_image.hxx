// Created by A. Aichert in Aug 2013

#ifndef __NRRD_IMAGE_HXX
#define __NRRD_IMAGE_HXX

#include "nrrd_image_view.hxx"
using namespace std;

namespace NRRD
{
	/// Simple image container with loading functoinality.
	template <typename T>
	class Image : public ImageView<T>
	{
	private:
		/// Not copy constructible
		Image(const Image&);
		/// Not copyable
		Image& operator=(const Image&);
	public:
		/// An invalid image
                Image() : ImageView<T>() {}

		/// An image from a .nrrd-file
                Image(const std::string& nrrd_file) : ImageView<T>(), alloc(0) { load(nrrd_file); }
		
		/// A 3D image of specific size. Optionally wraps an existing data pointer.
                Image(int w, int h, int d=1, T* dt=0x0) : ImageView<T>(), alloc(0) { set(w,h,d, dt); }

		/// A n-D image of specific size. Optionally wraps an existing data pointer.
                Image(int size[], int n, T* dt=0x0) : ImageView<T>(), alloc(0) { set(size,n,dt); }

		/// Create an image from file
		bool load(const std::string& nrrd_file)
		{
            if (alloc&&this->data!=0x0)
                    delete [] this->data;
            this->data=0x0;
            this->dim.clear();
            if (!NRRD::load<T>(nrrd_file,&this->data,&this->dim,&this->meta_info,&this->nrrd_header))
	return false;
            auto it=this->nrrd_header.find("spacings");
            if (it!=this->nrrd_header.end())
                    this->element_spacing=stringToVector<double>(it->second,' ');
            while (this->element_spacing.size()<this->dim.size()) this->element_spacing.push_back(1.0);

            auto it2 = this->nrrd_header.find("space directions");
            vector<string> space_dir_str;
            if (it2 != this->nrrd_header.end())
				space_dir_str = stringToVector<string>(it2->second, ' ');

			for (auto dir : space_dir_str) {
				std::vector<double> vect;

				std::stringstream ss(dir);

				//if (ss.peek() == 'n') {
				//	space_directions.push_back(vect);
				//	continue;
				//}
				if (ss.peek() == '(') {
					ss.ignore();

					double i;

					while (ss >> i)
					{
						vect.push_back(i);

						if (ss.peek() == ',')
							ss.ignore();
					}
				}
                this->space_dir.push_back(vect);
			}

                        auto it3 = this->nrrd_header.find("space origin");
			std::stringstream ss(it3->second);

			if (ss.peek() == '(') {
				ss.ignore();

				double i;

				while (ss >> i)
				{
                                        this->space_org.push_back(i);

					if (ss.peek() == ',')
						ss.ignore();
				}
			}
			return true;
		}
		
		/// Allocate or reference a 2D or 3D raw image.
		virtual void set(int w, int h, int d=1, T* dt=0x0)
		{
			int size[]={w,h,d};
			int n=d<2?2:3;
			set(size,n,dt);
		}

		/// Allocate or reference n-D image.
		virtual void set(const int size[], int n, T* dt=0x0)
		{
                        if (alloc&&this->data) {
                                delete [] this->data;
                                this->data=0x0;
			}
			alloc=false;
                        ImageView<T>::set(size,n,dt);
                        if (!this->data && this->length()>0)
			{	
				alloc=true;
                                this->data=new T[this->length()];
			}
		}

		/// Copy image data from another instance
		template <typename T2>
		void copyFrom(const NRRD::ImageView<T2>& img)
		{
                        this->dim.resize(img.dimension());
			for(int i=0;i<img.dimension();i++)
                                this->dim[i]=img.size(i);
			int n=img.length();
			#pragma omp parallel for
			for (int i=0;i<n;i++)
                                this->data[i]=img[i];
		}

		/// Prevents this class from calling delete [] on the data pointer.
		/// If return value is true, you HAVE TO delete the pionter yourself AFTER this object goes out of scope.
		/// Returns false if delete would not have been called anyway and caller MUST NOT call delete [] by himself.
		bool passOwnership()
		{
			if (alloc) return !(alloc=false);
			else return false;
		}

		/// Destructor. Deletes data pointer if the data has been allocated during construction.
		/// See also bool NRRD::Image<T>::passOwnership();
		~Image()
		{
                        if (alloc&&this->data!=0x0)
                                delete [] this->data;
		}

	private:
		/// True if destructor will call delete [].
		/// See also ::passOwnership()
		bool alloc;
	};

} // namespace NRRD

#endif // __NRRD_IMAGE_HXX
