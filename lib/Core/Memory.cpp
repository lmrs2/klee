//===-- Memory.cpp --------------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Memory.h"

#include "Context.h"
#include "MemoryManager.h"

#include "klee/Expr/ArrayCache.h"
#include "klee/Expr/Expr.h"
#include "klee/Internal/Support/ErrorHandling.h"
#include "klee/OptionCategories.h"
#include "klee/Solver/Solver.h"
#include "klee/util/BitArray.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <sstream>

using namespace llvm;
using namespace klee;

namespace {
  cl::opt<bool>
  UseConstantArrays("use-constant-arrays",
                    cl::desc("Use constant arrays instead of updates when possible (default=true)\n"),
                    cl::init(true),
                    cl::cat(SolvingCat));
}

/***/

int MemoryObject::counter = 0;

MemoryObject::~MemoryObject() {
  if (parent)
    parent->markFreed(this);
}

void MemoryObject::getAllocInfo(std::string &result) const {
  llvm::raw_string_ostream info(result);

  info << "MO" << id << "[" << size << "]";

  if (allocSite) {
    info << " allocated at ";
    if (const Instruction *i = dyn_cast<Instruction>(allocSite)) {
      info << i->getParent()->getParent()->getName() << "():";
      info << *i;
    } else if (const GlobalValue *gv = dyn_cast<GlobalValue>(allocSite)) {
      info << "global:" << gv->getName();
    } else {
      info << "value:" << *allocSite;
    }
  } else {
    info << " (no allocation info)";
  }
  
  info.flush();
}

/***/

ObjectState::ObjectState(const MemoryObject *mo)
  : copyOnWriteOwner(0),
    object(mo),
    concreteStore(new uint8_t[mo->size]),
    concreteMask(0),
    flushMask(0),
    undefinedMask(0),
    knownSymbolics(0),
    updates(0, 0),
    size(mo->size),
    readOnly(false) {
  if (!UseConstantArrays) {
    static unsigned id = 0;
    const Array *array =
        getArrayCache()->CreateArray("tmp_arr" + llvm::utostr(++id), size);
    updates = UpdateList(array, 0);
  }
  memset(concreteStore, 0, size);
}


ObjectState::ObjectState(const MemoryObject *mo, const Array *array)
  : copyOnWriteOwner(0),
    object(mo),
    concreteStore(new uint8_t[mo->size]),
    concreteMask(0),
    flushMask(0),
    undefinedMask(0),
    knownSymbolics(0),
    updates(array, 0),
    size(mo->size),
    readOnly(false) {
  makeSymbolic();
  memset(concreteStore, 0, size);
}

extern bool enteredDevMain;

ObjectState::ObjectState(const ObjectState &os) 
  : copyOnWriteOwner(0),
    object(os.object),
    concreteStore(new uint8_t[os.size]),
    concreteMask(os.concreteMask ? new BitArray(*os.concreteMask, os.size) : 0),
    flushMask(os.flushMask ? new BitArray(*os.flushMask, os.size) : 0),
    undefinedMask(os.undefinedMask ? new BitArray(*os.undefinedMask, os.size) : 0),
    knownSymbolics(0),
    updates(os.updates),
    size(os.size),
    readOnly(false) {
  assert(!os.readOnly && "no need to copy read only object?");
  if (os.knownSymbolics) {
    knownSymbolics = new ref<Expr>[size];
    for (unsigned i=0; i<size; i++)
      knownSymbolics[i] = os.knownSymbolics[i];
  }

  memcpy(concreteStore, os.concreteStore, size*sizeof(*concreteStore));
  if (undefinedMask) {
    if (!enteredDevMain) return;
    //errs() << "undefinedMask set in ObjectState() for " << this << "\n";

    // size_t i = 0;
    // for (i=0; i<size; ++i) {
    //   errs() << undefinedMask->get(i) << " ";
    // }
    // errs() << "\n";
  }
}

ObjectState::~ObjectState() {
  delete concreteMask;
  delete undefinedMask;
  delete flushMask;
  delete[] knownSymbolics;
  delete[] concreteStore;
}

ArrayCache *ObjectState::getArrayCache() const {
  assert(!object.isNull() && "object was NULL");
  return object->parent->getArrayCache();
}

/***/

const UpdateList &ObjectState::getUpdates() const {
  // Constant arrays are created lazily.
  if (!updates.root) {
    // Collect the list of writes, with the oldest writes first.
    
    // FIXME: We should be able to do this more efficiently, we just need to be
    // careful to get the interaction with the cache right. In particular we
    // should avoid creating UpdateNode instances we never use.
    unsigned NumWrites = updates.head.isNull() ? 0 : updates.head->getSize();
    std::vector< std::pair< ref<Expr>, ref<Expr> > > Writes(NumWrites);
    const auto *un = updates.head.get();
    for (unsigned i = NumWrites; i != 0; un = un->next.get()) {
      --i;
      Writes[i] = std::make_pair(un->index, un->value);
    }

    std::vector< ref<ConstantExpr> > Contents(size);

    // Initialize to zeros.
    for (unsigned i = 0, e = size; i != e; ++i)
      Contents[i] = ConstantExpr::create(0, Expr::Int8);

    // Pull off as many concrete writes as we can.
    unsigned Begin = 0, End = Writes.size();
    for (; Begin != End; ++Begin) {
      // Push concrete writes into the constant array.
      ConstantExpr *Index = dyn_cast<ConstantExpr>(Writes[Begin].first);
      if (!Index)
        break;

      ConstantExpr *Value = dyn_cast<ConstantExpr>(Writes[Begin].second);
      if (!Value)
        break;

      Contents[Index->getZExtValue()] = Value;
    }

    static unsigned id = 0;
    const Array *array = getArrayCache()->CreateArray(
        "const_arr" + llvm::utostr(++id), size, &Contents[0],
        &Contents[0] + Contents.size());
    updates = UpdateList(array, 0);

    // Apply the remaining (non-constant) writes.
    for (; Begin != End; ++Begin)
      updates.extend(Writes[Begin].first, Writes[Begin].second);
  }

  return updates;
}

void ObjectState::flushToConcreteStore(TimingSolver *solver,
                                       const ExecutionState &state) const {
  //errs() << "flushToConcreteStore\n";
  for (unsigned i = 0; i < size; i++) {
    if (isByteKnownSymbolic(i)) {
      ref<ConstantExpr> ce;
      ref<Expr> tmp = read8(i);
      assert(!tmp.isNull());
      bool success = solver->getValue(state, tmp, ce);
      if (!success)
        klee_warning("Solver timed out when getting a value for external call, "
                     "byte %p+%u will have random value",
                     (void *)object->address, i);
      else
        ce->toMemory(concreteStore + i);
    } else {
      if (enteredDevMain) {
        errs() << "undefined:" << isByteUndefined(i) << " concrete:" << isByteConcrete(i) << " isByteKnownSymbolic:" << isByteKnownSymbolic(i) << "\n";
        assert (0 && "what to do !?");
      }
    }
  }
}

void ObjectState::ignoreUndefined() {
  makeConcrete();
}

void ObjectState::makeConcrete() {
  delete undefinedMask;
  delete concreteMask;
  delete flushMask;
  delete[] knownSymbolics;
  undefinedMask = 0;
  concreteMask = 0;
  flushMask = 0;
  knownSymbolics = 0;
}

void ObjectState::makeSymbolic() {
  assert(updates.head.isNull() &&
         "XXX makeSymbolic of objects with symbolic values is unsupported");

  // XXX simplify this, can just delete various arrays I guess
  for (unsigned i=0; i<size; i++) {
    markByteSymbolic(i);
    setKnownSymbolic(i, 0);
    markByteFlushed(i);
  }
}

void ObjectState::initializeToZero() {
  makeConcrete();
  memset(concreteStore, 0, size);
}

void ObjectState::initializeToRandom() {  
  makeConcrete();
  for (unsigned i=0; i<size; i++) {
    // randomly selected by 256 sided die
    concreteStore[i] = 0xAB;
  }
}

void ObjectState::initializeToUndefined() {
  for (unsigned i=0; i<size; ++i)
    markByteFlushed(i); // this needs to be done before creating undefinedMask

  assert (!undefinedMask);
  undefinedMask = new BitArray(size, true);
}

/*
Cache Invariants
--
isByteKnownSymbolic(i) => !isByteConcrete(i)
isByteConcrete(i) => !isByteKnownSymbolic(i)
!isByteFlushed(i) => (isByteConcrete(i) || isByteKnownSymbolic(i))
 */

void ObjectState::fastRangeCheckOffset(ref<Expr> offset,
                                       unsigned *base_r,
                                       unsigned *size_r) const {
  *base_r = 0;
  *size_r = size;
}

void ObjectState::flushRangeForRead(unsigned rangeBase, 
                                    unsigned rangeSize) const {
  if (!flushMask) flushMask = new BitArray(size, true);
 
  for (unsigned offset=rangeBase; offset<rangeBase+rangeSize; offset++) {

    /*
    //assert (!isByteUndefined(offset)); // test
    if (isByteUndefined(offset)) {
      // an undefined byte is always flushed

      // errs() << "setting offset " << offset << " for " << this << "\n";
      // errs() << " for rangeBase " << rangeBase << " and  rangeSize " << rangeSize << "\n";
      
      //errs() << "ignoring flushRangeForRead\n";
      
      //undefinedOffset = offset;
      //return false;
      continue;
    }
    */

    if (!isByteFlushed(offset)) {
      assert (!isByteUndefined(offset));
      if (isByteConcrete(offset)) {
        updates.extend(ConstantExpr::create(offset, Expr::Int32),
                       ConstantExpr::create(concreteStore[offset], Expr::Int8));
      } else {
        if (!isByteKnownSymbolic(offset)) {
          errs() << "undefined:" << isByteUndefined(offset) << " concrete:" << isByteConcrete(offset) << " isByteKnownSymbolic:" << isByteKnownSymbolic(offset) << " for " << this << "\n";
        }
        assert(isByteKnownSymbolic(offset) && "invalid bit set in flushMask");
        updates.extend(ConstantExpr::create(offset, Expr::Int32),
                       knownSymbolics[offset]);
      }

      flushMask->unset(offset);
      // if (undefinedMask) // this code duplication sucks. We should be calling markByteFlushed() and markByteDefined(). This function changes state yet is marked const!?
      //   undefinedMask->unset(offset);
      //markByteFlushed(offset);
    }
  } 
}

void ObjectState::flushRangeForWrite(unsigned rangeBase, 
                                     unsigned rangeSize) {
  if (!flushMask) flushMask = new BitArray(size, true);

  for (unsigned offset=rangeBase; offset<rangeBase+rangeSize; offset++) {
    // if (isByteUndefined(offset))
    //   continue;
    if (!isByteFlushed(offset)) {
      assert (!isByteUndefined(offset));
      if (/*isByteUndefined(offset) ||*/ isByteConcrete(offset)) {
        updates.extend(ConstantExpr::create(offset, Expr::Int32),
                       ConstantExpr::create(concreteStore[offset], Expr::Int8));
        markByteSymbolic(offset);
      } else {
        assert(isByteKnownSymbolic(offset) && "invalid bit set in flushMask");
        updates.extend(ConstantExpr::create(offset, Expr::Int32),
                       knownSymbolics[offset]);
        setKnownSymbolic(offset, 0);
      }

      flushMask->unset(offset);
      // if (undefinedMask) // this code duplication sucks. We should be calling markByteFlushed() and markByteDefined()
      //   undefinedMask->unset(offset);
      //markByteFlushed(offset); // now this will be defined
    } else {
      // flushed bytes that are written over still need
      // to be marked out
      if (isByteUndefined(offset) || isByteConcrete(offset)) {
        markByteSymbolic(offset);
      } else if (isByteKnownSymbolic(offset)) {
        setKnownSymbolic(offset, 0);
      }
    }
  } 
}

bool ObjectState::isByteUndefined(unsigned offset, bool print) const {
  if (!enteredDevMain) return false;
  if (print)
    errs() << offset << " undefinedMask:" << undefinedMask << " " << undefinedMask->get(offset) << "\n";
  return undefinedMask && undefinedMask->get(offset);
}

bool ObjectState::isByteConcrete(unsigned offset) const {
  if (isByteUndefined(offset)) return false;
  return !concreteMask || concreteMask->get(offset);
}

bool ObjectState::isByteFlushed(unsigned offset) const {
  return flushMask && !flushMask->get(offset);
}

bool ObjectState::isByteKnownSymbolic(unsigned offset) const {
  if (isByteUndefined(offset)) return false;
  return knownSymbolics && knownSymbolics[offset].get();
}

void ObjectState::markByteDefined(unsigned offset) {
  //if (!enteredDevMain) return;
  // assert(undefinedMask);
  // if (!undefinedMask)
  //   undefinedMask = new BitArray(size, true);
  if (undefinedMask)
    undefinedMask->unset(offset);
  // if (enteredDevMain) {
  //   errs() << "ObjectState::markByteDefined:" << offset << " for " << this << "\n";
  // }
}

void ObjectState::markByteConcrete(unsigned offset) {
  if (concreteMask)
    concreteMask->set(offset);
  
  markByteDefined(offset);
}

void ObjectState::markByteSymbolic(unsigned offset) {
  if (!concreteMask)
    concreteMask = new BitArray(size, true);
  concreteMask->unset(offset);
  
  markByteDefined(offset);
}

void ObjectState::markByteUnflushed(unsigned offset) {
  if (flushMask)
    flushMask->set(offset);
}

void ObjectState::markByteFlushed(unsigned offset) {
  // if (enteredDevMain)
  //   errs() << "markByteFlushed " << offset << "\n";
  if (!flushMask) {
    flushMask = new BitArray(size, false);
  } else {
    flushMask->unset(offset);
  }
  markByteDefined(offset);
}

void ObjectState::setKnownSymbolic(unsigned offset, 
                                   Expr *value /* can be null */) {
  // if (enteredDevMain)
  //   errs() << "setKnownSymbolic " << offset << "\n";
  if (knownSymbolics) {
    knownSymbolics[offset] = value;
  } else {
    if (value) {
      knownSymbolics = new ref<Expr>[size];
      knownSymbolics[offset] = value;
    }
  }
  markByteDefined(offset);
}

/***/
extern bool enteredDevMain;
ref<Expr> ObjectState::read8(unsigned offset) const {
  // if (enteredDevMain) {
  //   errs() << "ObjectState::read8:" << offset << " for " << this << "\n";
  //   errs() << "undefined:" << isByteUndefined(offset) << " concrete:" << isByteConcrete(offset) << " isByteKnownSymbolic:" << isByteKnownSymbolic(offset) << " for " << this << "\n";
  // }

  if (isByteUndefined(offset)) {
    return nullptr;
  } else if (isByteConcrete(offset)) {
    return ConstantExpr::create(concreteStore[offset], Expr::Int8);
  } else if (isByteKnownSymbolic(offset)) {
    return knownSymbolics[offset];
  } else {
    assert(isByteFlushed(offset) && "unflushed byte without cache value");
    //assert(0 && "TODO concolic");
    return ReadExpr::create(getUpdates(), 
                            ConstantExpr::create(offset, Expr::Int32));
  }    
}

// TODO: symbolic offsets
ref<Expr> ObjectState::read8(ref<Expr> offset, unsigned & undefinedOffset) const {
  // if (enteredDevMain) {
  //   errs() << "ObjectState::read8-2:" << offset << " for " << this << "\n";
  // }
  assert(!isa<ConstantExpr>(offset) && "constant offset passed to symbolic read8");
  unsigned base, size;
  fastRangeCheckOffset(offset, &base, &size);

  // check for undefined bytes
  // for (unsigned i=base; i<base+size; i++) {
  //   if (isByteUndefined(i)) {
  //     undefinedOffset = i;
  //     return nullptr;
  //   }
  // }
  flushRangeForRead(base, size);

  if (size>4096) {
    std::string allocInfo;
    object->getAllocInfo(allocInfo);
    klee_warning_once(0, "flushing %d bytes on read, may be slow and/or crash: %s", 
                      size,
                      allocInfo.c_str());
  }
  
  return ReadExpr::create(getUpdates(), ZExtExpr::create(offset, Expr::Int32));
}

void ObjectState::write8(unsigned offset, uint8_t value) {
  // if (enteredDevMain)
  //   errs() << "ObjectState::write8:" << offset << ":" << value << "\n";
  //assert(read_only == false && "writing to read-only object!");
  concreteStore[offset] = value;
  setKnownSymbolic(offset, 0);

  markByteConcrete(offset);
  markByteUnflushed(offset);
}

void ObjectState::write8(unsigned offset, ref<Expr> value) {
  // if (enteredDevMain)
  //   errs() << "ObjectState::write8:2 " << offset << ":" << value << "\n";
  // can happen when ExtractExpr special cases
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(value)) {
    write8(offset, (uint8_t) CE->getZExtValue(8));
  } else {
    setKnownSymbolic(offset, value.get());
      
    markByteSymbolic(offset);
    markByteUnflushed(offset);
  }
}

void ObjectState::write8(ref<Expr> offset, ref<Expr> value) {
  assert(!isa<ConstantExpr>(offset) && "constant offset passed to symbolic write8");
  // if (enteredDevMain)
  //   errs() << "ObjectState::write8:3 " << offset << ":" << value << "\n";

  unsigned base, size;
  fastRangeCheckOffset(offset, &base, &size);
  flushRangeForWrite(base, size);

  if (size>4096) {
    std::string allocInfo;
    object->getAllocInfo(allocInfo);
    klee_warning_once(0, "flushing %d bytes on read, may be slow and/or crash: %s", 
                      size,
                      allocInfo.c_str());
  }
  
  updates.extend(ZExtExpr::create(offset, Expr::Int32), value);
}

/***/

ref<Expr> ObjectState::read(ref<Expr> offset, Expr::Width width, unsigned & undefinedOffset) const {
  // Truncate offset to 32-bits.
  offset = ZExtExpr::create(offset, Expr::Int32);

// if (enteredDevMain) 
//     errs() << "ObjectState::read(expr):" << offset << " for " << this << "\n";
  // Check for reads at constant offsets.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(offset))
    return read(CE->getZExtValue(32), width, undefinedOffset);

  // Treat bool specially, it is the only non-byte sized write we allow.
  if (width == Expr::Bool) {
    // if (enteredDevMain)
    //   errs() << "is a bool\n";
    //ref<Expr> tmp = nullptr;
    //if ((tmp = read8(offset, undefinedOffset)).isNull())
    //  return nullptr;

    // TODO: remove undefinedoffset
    return ExtractExpr::create(read8(offset, undefinedOffset), 0, Expr::Bool);
  }

  // Otherwise, follow the slow general case.
  unsigned NumBytes = width / 8;
  assert(width == NumBytes * 8 && "Invalid read size!");
  ref<Expr> Res(0);
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    ref<Expr> Byte = read8(AddExpr::create(offset, 
                                           ConstantExpr::create(idx, 
                                                                Expr::Int32)),
                          undefinedOffset);
    // if (Byte.isNull()) {
    //   // assert(undefinedOffset >= idx);
    //   // undefinedOffset -= idx; // Mhh!?
    //   //
    //   //errs() << " HEY:" << AddExpr::create(offset, 
    //   //                                     ConstantExpr::create(idx, 
    //   //                                                          Expr::Int32)) << "\n";
    //   //errs() << "offset:" << offset << " i:" << i << "\n";
    //   //assert(0 && "CHECK");
    //   return nullptr;
    // }
    Res = i ? ConcatExpr::create(Byte, Res) : Byte;
  }

  return Res;
}

ref<Expr> ObjectState::read(unsigned offset, Expr::Width width, unsigned & undefinedOffset) const {
  // if (enteredDevMain) 
//     errs() << "ObjectState::read(concrete):" << offset << " for " << this << "\n";
  // Treat bool specially, it is the only non-byte sized write we allow.
  if (width == Expr::Bool) {
    ref<Expr> tmp = nullptr;
    if ((tmp = read8(offset)).isNull()) {
      undefinedOffset = offset;
      return nullptr;
    }
    return ExtractExpr::create(tmp, 0, Expr::Bool);
  }

  // Otherwise, follow the slow general case.
  unsigned NumBytes = width / 8;
  assert(width == NumBytes * 8 && "Invalid width for read size!");
  ref<Expr> Res(0);
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    ref<Expr> Byte = read8(offset + idx);
    if (Byte.isNull()) {
      undefinedOffset = offset + idx;
      return nullptr;
    }
    Res = i ? ConcatExpr::create(Byte, Res) : Byte;
  }

  return Res;
}

void ObjectState::write(ref<Expr> offset, ref<Expr> value) {
  // if (enteredDevMain)
  //   errs() << "ObjectState::write:exp[" << offset << "] = " << value << " for " << this << "\n";
  // Truncate offset to 32-bits.
  offset = ZExtExpr::create(offset, Expr::Int32);

  // Check for writes at constant offsets.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(offset)) {
    write(CE->getZExtValue(32), value);
    return;
  }

  // Treat bool specially, it is the only non-byte sized write we allow.
  Expr::Width w = value->getWidth();
  if (w == Expr::Bool) {
    write8(offset, ZExtExpr::create(value, Expr::Int8));
    return;
  }

  // Otherwise, follow the slow general case.
  unsigned NumBytes = w / 8;
  assert(w == NumBytes * 8 && "Invalid write size!");
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);

    // if (enteredDevMain) {
    //   errs() << "i = " << i << "\n";
    //   errs() << "expr:" << AddExpr::create(offset, ConstantExpr::create(idx, Expr::Int32)) << " " <<
    //        ExtractExpr::create(value, 8 * i, Expr::Int8) << "\n";
    // }

    write8(AddExpr::create(offset, ConstantExpr::create(idx, Expr::Int32)),
           ExtractExpr::create(value, 8 * i, Expr::Int8));
  }

  // if (enteredDevMain)
  //   errs() << "finished write()\n";
}

void ObjectState::write(unsigned offset, ref<Expr> value, bool print) {
  // if (enteredDevMain || print)
  //   errs() << "ObjectState::write:con[" << offset << "] =" << value.isNull() << "\n";
  // Check for writes of constant values.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(value)) {
    // if (print)
    //   errs() << " constant\n";
    Expr::Width w = CE->getWidth();
    if (w <= 64 && klee::bits64::isPowerOfTwo(w)) {
      uint64_t val = CE->getZExtValue();
      switch (w) {
      default: assert(0 && "Invalid write size!");
      case  Expr::Bool:
      case  Expr::Int8:  write8(offset, val); return;
      case Expr::Int16: write16(offset, val); return;
      case Expr::Int32: write32(offset, val); return;
      case Expr::Int64: write64(offset, val); return;
      }
    }
  }

  // Treat bool specially, it is the only non-byte sized write we allow.
  Expr::Width w = value->getWidth();
  if (w == Expr::Bool) {
    // if (print)
    //   errs() << " write8\n";
    write8(offset, ZExtExpr::create(value, Expr::Int8));
    return;
  }

  // Otherwise, follow the slow general case.
  unsigned NumBytes = w / 8;
  assert(w == NumBytes * 8 && "Invalid write size!");
  // if (print)
  //     errs() << " loop write8\n";
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    write8(offset + idx, ExtractExpr::create(value, 8 * i, Expr::Int8));
  }
} 

void ObjectState::write16(unsigned offset, uint16_t value) {
  unsigned NumBytes = 2;
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    write8(offset + idx, (uint8_t) (value >> (8 * i)));
  }
}

void ObjectState::write32(unsigned offset, uint32_t value) {
  unsigned NumBytes = 4;
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    write8(offset + idx, (uint8_t) (value >> (8 * i)));
  }
}

void ObjectState::write64(unsigned offset, uint64_t value) {
  unsigned NumBytes = 8;
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    write8(offset + idx, (uint8_t) (value >> (8 * i)));
  }
}

void ObjectState::print() const {
  llvm::errs() << "-- ObjectState --\n";
  llvm::errs() << "\tMemoryObject ID: " << object->id << "\n";
  llvm::errs() << "\tRoot Object: " << updates.root << "\n";
  llvm::errs() << "\tSize: " << size << "\n";

  llvm::errs() << "\tBytes:\n";
  for (unsigned i=0; i<size; i++) {
    llvm::errs() << "\t\t["<<i<<"]"
               << " undefined? " << isByteUndefined(i)
               << " concrete? " << isByteConcrete(i)
               << " known-sym? " << isByteKnownSymbolic(i)
               << " flushed? " << isByteFlushed(i) << " = ";
    ref<Expr> e = read8(i);
    if (e.isNull())
      llvm::errs() << "<NA>" << "\n";
    else  
      llvm::errs() << e << "\n";
  }

  llvm::errs() << "\tUpdates:\n";
  for (const auto *un = updates.head.get(); un; un = un->next.get()) {
    llvm::errs() << "\t\t[" << un->index << "] = " << un->value << "\n";
  }
}
