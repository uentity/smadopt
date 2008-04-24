#ifndef DETERM_ANNEALING_H_
#define DETERM_ANNEALING_H_

#include "common.h"

namespace DA {

class determ_annealing
{
public:
	determ_annealing();
	~determ_annealing();
protected:
private:
	class da_impl;
	smart_ptr< da_impl > pimpl_;
};

}	//end of namespace DA

#endif // DETERM_ANNEALING_H_
