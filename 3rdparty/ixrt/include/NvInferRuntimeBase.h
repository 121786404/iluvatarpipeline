// Created on 2023/11/30.
#pragma once
#include <cstdint>

#include "NvInferVersion.h"
namespace nvinfer1 {
//!
//! \enum ErrorCode
//!
//! \brief Error codes that can be returned by IxRT during execution.
//!
enum class ErrorCode : int32_t {
    //!
    //! Execution completed successfully.
    //!
    kSUCCESS = 0,

    //!
    //! An error that does not fall into any other category.
    //!
    kUNSPECIFIED_ERROR = 1,

    //!
    //! An non-recoverable error happened in IxRT
    //!
    kINTERNAL_ERROR = 2,

    //!
    //! An invalid argument was given to the interface
    //!
    kINVALID_ARGUMENT = 3,

    //!
    //! Invalid config which violates IxRT contract.
    //!
    kINVALID_CONFIG = 4,

    //!
    //! Failed to allocate memory for either device or host
    //!
    kFAILED_ALLOCATION = 5,

    //!
    //! One, or more, of the components that IxRT relies on did not initialize correctly.
    //! This is a system setup issue.
    //!
    kFAILED_INITIALIZATION = 6,

    //!
    //! An error occurred during execution that caused IxRT to end prematurely.
    //! This is either an execution error or a memory error.
    //!
    kFAILED_EXECUTION = 7,

    //!
    //! Either a data corruption error, an input error, or a range error.
    //!
    kFAILED_COMPUTATION = 8,

    //!
    //! Internal invalid state
    //!
    kINVALID_STATE = 9,

    //!
    //! An error occurred due to the network not being supported.
    //!
    kUNSUPPORTED_STATE = 10,

};

//!
//! \class IErrorRecorder
//!
class IErrorRecorder {
   public:
    //!
    //! A typedef of a C-style string for reporting error descriptions.
    //!
    using ErrorDesc = char const*;

    //!
    //! The length limit for an error description, excluding the '\0' string terminator.
    //!
    static constexpr size_t kMAX_DESC_LENGTH{127U};

    //!
    //! A typedef of a 32bit integer for reference counting.
    //!
    using RefCount = int32_t;

    IErrorRecorder() = default;
    virtual ~IErrorRecorder() noexcept = default;

    // Public API used to retrieve information from the error recorder.

    //!
    //! \brief Return the number of errors
    //!
    //! Determines the number of errors that occurred between the current point in execution
    //!
    //! \return Returns the number of errors detected, or 0 if there are no errors.
    //!
    //! \see clear
    //!
    virtual int32_t getNbErrors() const noexcept = 0;

    //!
    //! \brief Returns the ErrorCode enumeration.
    //!
    //! \param errorIdx A 32-bit integer that indexes into the error array.
    //!
    //! The errorIdx specifies what error code from 0 to getNbErrors()-1 that the application
    //! wants to analyze and return the error code enum.
    //!
    //! \return Returns the enum corresponding to errorIdx.
    //!
    //! \see getErrorDesc, ErrorCode
    //!
    virtual ErrorCode getErrorCode(int32_t errorIdx) const noexcept = 0;

    //!
    //! \brief Returns a null-terminated C-style string description of the error.
    //!
    //! \param errorIdx A 32-bit integer that indexes into the error array.
    //!
    //! \return Returns a string representation of the error along with a description of the error.
    //!
    //! \see getErrorCode
    //!
    virtual ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept = 0;

    //!
    //! \brief Determine if the error stack has overflowed.
    //!
    //! \return true if errors have been dropped due to overflowing the error stack.
    //!
    virtual bool hasOverflowed() const noexcept = 0;

    //!
    //! \brief Clear the error stack on the error recorder.
    //!
    //! \see getNbErrors
    //!
    virtual void clear() noexcept = 0;

    // API used by IxRT to report Error information to the application.

    //!
    //! \brief Report an error to the error recorder with the corresponding enum and description.
    //!
    //! \param val The error code enum that is being reported.
    //! \param desc The string description of the error.
    //!
    //! \return True if the error is determined to be fatal and processing of the current function must end.
    //!
    virtual bool reportError(ErrorCode val, ErrorDesc desc) noexcept = 0;

    //!
    //! \brief Increments the refcount for the current ErrorRecorder.
    //!
    //! \return The reference counted value after the increment completes.
    //!
    virtual RefCount incRefCount() noexcept = 0;

    //!
    //! \brief Decrements the refcount for the current ErrorRecorder.
    //!
    //! \return The reference counted value after the decrement completes.
    //!
    virtual RefCount decRefCount() noexcept = 0;

   protected:
    // @cond SuppressDoxyWarnings
    IErrorRecorder(IErrorRecorder const&) = default;
    IErrorRecorder(IErrorRecorder&&) = default;
    IErrorRecorder& operator=(IErrorRecorder const&) & = default;
    IErrorRecorder& operator=(IErrorRecorder&&) & = default;
    // @endcond
};  // class IErrorRecorder

//!
//! \enum TensorIOMode
//!
//! \brief Definition of tensor IO Mode.
//!
enum class TensorIOMode : int32_t {
    //! Tensor is not an input or output.
    kNONE = 0,

    //! Tensor is input to the engine.
    kINPUT = 1,

    //! Tensor is output by the engine.
    kOUTPUT = 2
};

//!
//! \enum APILanguage
//!
//! \brief Programming language used in the implementation of a TRT interface
//!
enum class APILanguage : int32_t { kCPP = 0, kPYTHON = 1 };

using InterfaceKind = char const*;

//!
//! \class InterfaceInfo
//!
//! \brief Version information associated with a TRT interface
//!
class InterfaceInfo {
   public:
    InterfaceKind kind;
    int32_t major;
    int32_t minor;
};

//!
//! \class IVersionedInterface
//!
//! \brief An Interface class for version control.
//!
class IVersionedInterface {
   public:
    //!
    //! \brief The language used to build the implementation of this Interface.
    //!
    //! Applications must not override this method.
    //!
    virtual APILanguage getAPILanguage() const noexcept { return APILanguage::kCPP; }

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    virtual InterfaceInfo getInterfaceInfo() const noexcept = 0;

    virtual ~IVersionedInterface() noexcept = default;

   protected:
    IVersionedInterface() = default;
    IVersionedInterface(IVersionedInterface const&) = default;
    IVersionedInterface(IVersionedInterface&&) = default;
    IVersionedInterface& operator=(IVersionedInterface const&) & = default;
    IVersionedInterface& operator=(IVersionedInterface&&) & = default;
};

class IStreamReader : public IVersionedInterface {
   public:
    //!
    //! TensorRT never calls the destructor for an IStreamReader defined by the
    //! application.
    //!
    ~IStreamReader() override = default;
    IStreamReader() = default;

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override { return InterfaceInfo{"IStreamReader", 1, 0}; }

    //!
    //! \brief Read the next number of bytes in the stream.
    //!
    //! \param destination The memory to write to
    //! \param nbBytes The number of bytes to read
    //!
    //! \returns The number of bytes read. Negative values will be considered an automatic error.
    //!
    virtual int64_t read(void* destination, int64_t nbBytes) = 0;

   protected:
    IStreamReader(IStreamReader const&) = default;
    IStreamReader(IStreamReader&&) = default;
    IStreamReader& operator=(IStreamReader const&) & = default;
    IStreamReader& operator=(IStreamReader&&) & = default;
};

}  // namespace nvinfer1
