#
# THIS CODE IS COPIED FROM PYTO:
# https://github.com/vladanl/Pyto/tree/master/pyto/io
# THIS IS TO AVOID THE INSTALLATION ISSUES AND NO PYPI AVAILABLITY
# PLASE CITE ACCORDINGLY
#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, December 2021

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object

# from past.utils import old_div
from past.builtins import basestring

import sys
import struct
import re
import os.path
import logging

# import warnings
from copy import copy
import io
from io import open

import numpy

from . import microscope_db


class FileTypeError(IOError):
    """
    Exception raised when nonexistent file type is given.

    Attributes
    ----------
    requested : str
        The requested file type/format.
    defined : dict
        Dictionary of defined file formats with their extensions.

    """
    def __init__(self, requested, defined):
        """
        Initialize FileTypeError.

        Parameters
        ----------
        requested : str
            The requested file type/format.
        defined : dict
            Dictionary of defined file formats with their extensions.
        """
        self.requested = requested
        self.defined = defined

    def __str__(self):
        """
        Returns a formatted error message string.

        Returns
        -------
        str
            The error message string.
        """
        msg = (
            "Defined file formats are: \n\t"
            + str(list(set(self.defined.values())))
            + "\nand defined extensions are: \n\t"
            + str(set(self.defined.keys()))
        )
        if self.requested is None:
            msg = msg + " File format not understood. "
        else:
            msg = msg + " File format: " + self.requested + " doesn't exist. "
        return msg


class ImageIO(object):
    """
    Reads and writes EM image files in em, mrc and raw formats.

    Attributes
    ----------
    fileName : str
        The name of the file being processed.
    file_ : file instance
        File instance if it's already open.

    Examples
    --------
    Reading an image file:

    >>> myImage = ImageIO()
    >>> myImage.read(file='my_file.em')

    Writing an image:

    >>> myImage = ImageIO()
    >>> myImage.write(file='my_file.em', data=my_array, header=my_header)
    """
    # determine machine byte order
    byte_order = sys.byteorder
    if byte_order == "big":
        machineByteOrder = ">"
    elif byte_order == "little":
        machineByteOrder = "<"
    else:
        machineByteOrder = "<"
        logging.warning(
            "Machine byte order could not be determined, set to "
            + " '<' (little endian)."
        )

    def __init__(self, file=None):
        """
        Initializes ImageIO object.

        Parameters
        ----------
        file : str or file instance, optional
            Either the file name or an already opened file instance.
        """

        # initialize attributes
        self.byteOrder = None
        self.defaultArrayOrder = "C"
        self.arrayOrder = None
        self.dataType = None
        self.shape = None
        self.data = None
        self.axisOrder = None
        self.length = None
        self.pixel = None
        self.fileFormat = None

        self.mrcHeader = None
        self.emHeader = None
        self.rawHeader = None
        self.header = None
        self.rawHeaderSize = None

        # parse arguments
        if file is not None:
            if isinstance(file, basestring):
                self.fileName = file
            elif isinstance(file, file):
                self.file_ = file

        return

    ##########################################################
    #
    # General image file read and write
    #
    #########################################################

    # File formats and extensions
    fileFormats = {
        "em": "em",
        "EM": "em",
        "raw": "raw",
        "dat": "raw",
        "RAW": "raw",
        "mrc": "mrc",
        "rec": "mrc",
        "mrcs": "mrc",
    }

    def read(
        self,
        file=None,
        fileFormat=None,
        byteOrder=None,
        dataType=None,
        arrayOrder=None,
        shape=None,
        memmap=False,
    ):
        """
        Reads image file in em, mrc or raw data formats.

        Parameters
        ----------
        file : str, optional
            File name.
        fileFormat : {'em', 'mrc', 'raw'}, optional
            The format of the file.
        byteOrder : {'<', '>'}, optional
            Byte order: '<' for little-endian and '>' for big-endian.
        dataType : str, optional
            Data type like 'int8', 'int16', etc.
        arrayOrder : {'C', 'F'}, optional
            Array order: 'C' for z-axis fastest, 'F' for x-axis fastest.
        shape : tuple, optional
            Shape of the data in the format (x_dim, y_dim, z_dim).
        memmap : bool, optional
            If True, data is read to a memory map.

        Returns
        -------
        None

        Raises
        ------
        FileTypeError
            If the specified file format is not recognized.
        """
        # determine the file format
        self.setFileFormat(file_=file, fileFormat=fileFormat)
        if self.fileFormat is None:
            raise FileTypeError(requested=self.fileFormat, defined=self.fileFormats)

        # call the appropriate read method
        if self.fileFormat == "em":
            self.readEM(
                file=file,
                byteOrder=byteOrder,
                shape=shape,
                dataType=dataType,
                arrayOrder=arrayOrder,
                memmap=memmap,
            )
            self.header = self.emHeader
        elif self.fileFormat == "mrc":
            self.readMRC(
                file=file,
                byteOrder=byteOrder,
                shape=shape,
                dataType=dataType,
                arrayOrder=arrayOrder,
                memmap=memmap,
            )
            self.header = self.mrcHeader
        elif self.fileFormat == "raw":
            self.readRaw(
                file=file,
                byteOrder=byteOrder,
                dataType=dataType,
                arrayOrder=arrayOrder,
                shape=shape,
                memmap=memmap,
            )
            self.header = self.rawHeader
        else:
            raise FileTypeError(requested=self.fileFormat, defined=self.fileFormats)

        return

    def readHeader(self, file=None, fileFormat=None, byteOrder=None):
        """
        Reads the header of an image file in em, mrc or raw data formats.

        Parameters
        ----------
        file : str, optional
            File name.
        fileFormat : {'em', 'mrc', 'raw'}, optional
            The format of the file.
        byteOrder : {'<', '>'}, optional
            Byte order: '<' for little-endian and '>' for big-endian.

        Returns
        -------
        None

        Raises
        ------
        FileTypeError
            If the specified file format is not recognized.
        """

        # determine the file format
        self.setFileFormat(file_=file, fileFormat=fileFormat)
        if self.fileFormat is None:
            raise FileTypeError(requested=self.fileFormat, defined=self.fileFormats)

        # call the appropriate read method
        if self.fileFormat == "em":
            self.readEMHeader(file=file, byteOrder=byteOrder)
        elif self.fileFormat == "mrc":
            self.readMRCHeader(file=file, byteOrder=byteOrder)
        elif self.fileFormat == "raw":
            self.rawHeader = ""
        else:
            raise FileTypeError(requested=self.fileFormat, defined=self.fileFormats)

        return

    def write(
        self,
        file=None,
        data=None,
        fileFormat=None,
        byteOrder=None,
        dataType=None,
        arrayOrder=None,
        shape=None,
        length=None,
        pixel=None,
        header=None,
        extended=None,
        casting="unsafe",
    ):
        """
        Writes image file with specified header and data.

        Parameters
        ----------
        file : str, optional
            File name.
        data : ndarray
            Image data.
        fileFormat : {'em', 'mrc', 'raw'}, optional
            The format of the file.
        byteOrder : {'<', '>'}, optional
            Byte order: '<' for little-endian and '>' for big-endian.
        dataType : str, optional
            Data type like 'int8', 'int16', etc.
        arrayOrder : {'C', 'F'}, optional
            Array order: 'C' for z-axis fastest, 'F' for x-axis fastest.
        shape : tuple, optional
            Shape of the data in the format (x_dim, y_dim, z_dim).
        length : list or ndarray, optional
            Length in each dimension in nm (used only for mrc format).
        pixel : float, optional
            Pixel size in nm (used only for mrc format if length is None).
        header : list, optional
            Image header.
        extended : str, optional
            Extended header string (only for mrc format).
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Type of data casting that may occur.

        Returns
        -------
        None

        Raises
        ------
        FileTypeError
            If the specified file format is not recognized.
        """

        # determine the file format
        self.setFileFormat(file_=file, fileFormat=fileFormat)
        if self.fileFormat is None:
            raise FileTypeError(requested=self.fileFormat, defined=self.fileFormats)

        # just in case dataType is given as numpy.dtype
        if isinstance(dataType, numpy.dtype):
            dataType = str(dataType)

        # call the appropriate write method
        if self.fileFormat == "em":
            self.writeEM(
                file=file,
                data=data,
                header=header,
                byteOrder=byteOrder,
                dataType=dataType,
                arrayOrder=arrayOrder,
                shape=shape,
                casting=casting,
            )
        elif self.fileFormat == "mrc":
            self.writeMRC(
                file=file,
                data=data,
                header=header,
                byteOrder=byteOrder,
                dataType=dataType,
                arrayOrder=arrayOrder,
                shape=shape,
                length=length,
                pixel=pixel,
                extended=extended,
                casting=casting,
            )
        elif self.fileFormat == "raw":
            self.writeRaw(
                file=file,
                data=data,
                header=header,
                byteOrder=byteOrder,
                dataType=dataType,
                arrayOrder=arrayOrder,
                shape=shape,
                casting=casting,
            )
        else:
            raise FileTypeError(requested=self.fileFormat, defined=self.fileFormats)

        return self.file_

    ###########################################################
    #
    # EM format
    #
    ###########################################################

    # EM file format properties
    em = {
        "headerSize": 512,
        "headerFormat": "4b 3i 80s 40i 20s 8s 228s",
        "defaultByteOrder": machineByteOrder,
        #'arrayOrder': 'FORTRAN'
        "arrayOrder": "F",
    }
    emHeaderFields = (
        "machine",
        "newOS9",
        "noHeader",
        "dataTypeCode",
        "lengthX",
        "lengthY",
        "lengthZ",
        "comment",
        "voltage",
        "cs",
        "aperture",
        "magnification",
        "postmagnification",
        "exposureTime",
        "_pixelsize",
        "emCode",
        "ccdPixelsize",
        "ccdLength",
        "defocus",
        "astigmatism",
        "astigmatismAngle",
        "focusIncrement",
        "countsPerelectron",
        "intensity",
        "energySlitWidth",
        "energyOffset",
        "_tiltAngle",
        "tiltAxis",
        "field_21",
        "field_22",
        "field_23",
        "markerX",
        "markerY",
        "resolution",
        "density",
        "contrast",
        "field_29",
        "massCentreX",
        "massCentreY",
        "massCentreZ",
        "height",
        "field_34",
        "widthDreistrahlbereich",
        "widthAchromRing",
        "lambda",
        "deltaTheta",
        "field_39",
        "field_40",
        "username",
        "date",
        "userdata",
    )
    emDefaultHeader = (
        [6, 0, 0, 0, 1, 1, 1, 80 * b" "]
        + numpy.zeros(40, "int8").tolist()
        + [20 * b" ", 8 * b" ", 228 * b" "]
    )
    emDefaultShape = [0, 0, 0]
    # emDefaultDataType = 0
    emByteOrderTab = {5: ">", 6: "<"}  # Mac  # PC (Intel)
    emByteOrderTabInv = dict(
        list(zip(list(emByteOrderTab.values()), list(emByteOrderTab.keys())))
    )
    emDataTypeTab = {
        1: "uint8",
        2: "uint16",
        4: "int32",
        5: "float32",
        8: "complex64",
        9: "float64",
    }
    emDataTypeTabInv = dict(
        list(zip(list(emDataTypeTab.values()), list(emDataTypeTab.keys())))
    )

    def readEM(
        self,
        file=None,
        byteOrder=None,
        dataType=None,
        arrayOrder=None,
        shape=None,
        memmap=False,
    ):
        """
        Reads EM file format.

        Parameters
        ----------
        file : str, optional
            Path to the file to read.
        byteOrder : {None, '<', '>'}, optional
            Byte order for reading the file. '<' means little endian and '>' means
            big endian.
        dataType : str, optional
            Data type to interpret the data.
        arrayOrder : str, optional
            Order in which to read the array.
        shape : tuple of int, optional
            Shape of the data array.
        memmap : bool, default False
            If true, use memory mapping to read data.
        """
        # open the file if needed
        self.checkFile(file=file, mode="rb")

        # set defaults
        self.arrayOrder = ImageIO.em["arrayOrder"]

        # parse arguments
        if byteOrder is not None:
            self.byteOrder = byteOrder
        if dataType is not None:
            self.dataType = dataType
        if arrayOrder is not None:
            self.arrayOrder = arrayOrder
        if shape is not None:
            self.shape = shape

        # use defaults if needed
        if self.byteOrder is None:
            self.byteOrder = ImageIO.em["defaultByteOrder"]
        if self.arrayOrder is None:
            self.arrayOrder = ImageIO.em["arrayOrder"]

        # read the header
        self.readEMHeader(file=self.file_)

        # read the data
        self.readData(shape=shape, memmap=memmap)

        return

    def readEMHeader(self, file=None, byteOrder=None):
        """
        Reads the header of an EM file.

        Parameters
        ----------
        file : str, optional
            Path to the file to read header from.
        byteOrder : {None, '<', '>'}, optional
            Byte order for reading the file. '<' means little endian and '>' means
            big endian.
        """
        # open the file if needed
        self.checkFile(file=file, mode="rb")

        # read the header
        self.headerString = self.file_.read(ImageIO.em["headerSize"])

        # determine byte order
        if byteOrder is not None:  # explicit byte order
            self.byteOrder = byteOrder
        else:  # determine byte order form the file
            (self.machine, tmp) = struct.unpack("b 511s", self.headerString)
            self.byteOrder = ImageIO.emByteOrderTab[self.machine]
        format = self.byteOrder + ImageIO.em["headerFormat"]

        # unpack the header with the right byte order
        self.emHeader = list(struct.unpack(format, self.headerString))

        # parse data type and shape (important)
        self.dataType = ImageIO.emDataTypeTab[self.emHeader[3]]
        self.shape = self.emHeader[4:7]

        # parse the rest of the header
        for attr, val in zip(ImageIO.emHeaderFields, self.emHeader):
            self.__dict__[attr] = val

        return

    def writeEM(
        self,
        file=None,
        header=None,
        byteOrder=None,
        shape=None,
        dataType=None,
        arrayOrder=None,
        data=None,
        casting="unsafe",
    ):
        """
        Writes data to an EM file format.

        Parameters
        ----------
        file : str, optional
            Path to the file to write to.
        header : list, optional
            Header information for the file.
        byteOrder : {None, '<', '>'}, optional
            Byte order for writing the file. '<' means little endian and '>' means
            big endian.
        shape : tuple of int, optional
            Shape of the data array.
        dataType : str, optional
            Data type of the data to write.
        arrayOrder : str, optional
            Order in which to write the array.
        data : ndarray, optional
            Data array to write.
        casting : str, default "unsafe"
            Casting rule for numpy. Check numpy documentation for more details.
        """

        # open the file if needed
        self.checkFile(file=file, mode="wb")

        # set emHeader
        if header is not None:
            self.emHeader = header

        # byteOrder: use the argument, self.byteOrder, or the default value
        if byteOrder is not None:
            self.byteOrder = byteOrder
        if self.byteOrder is None:
            # if self.emHeader is not None:
            #    self.byteOrder = ImageIO.emByteOrderTab[self.emHeader[0]]
            # else:
            self.byteOrder = ImageIO.em["defaultByteOrder"]

        # arrayOrder: use the argument, self.arrayOrder, or the default value
        if arrayOrder is not None:
            self.arrayOrder = arrayOrder
        if self.arrayOrder is None:
            self.arrayOrder = ImageIO.em["arrayOrder"]

        # data: use the argument or the self.data, set self.shape and
        # self.dataType
        if data is not None:
            self.setData(data, shape=shape)

        # dataType: use the argument or self.dataType
        if dataType is not None:
            self.dataType = dataType
        # if self.dataType is None:
        #    self.dataType = self.emDefaultDataType
        # if self.dataType is None:
        # if self.emHeader is not None:
        #    self.dataType = ImageIO.emDataTypeTab[self.emHeader[3]]

        # convert data to another dtype if needed
        wrong_data_type = False
        try:
            if self.dataType == "uint8":
                if self.data.dtype.name != "uint8":
                    self.data = self.data.astype(dtype="uint8", casting=casting)
            elif self.dataType == "uint16":
                if self.data.dtype.name != "uint16":
                    self.data = self.data.astype(dtype="uint16", casting=casting)
            elif self.dataType == "int32":
                if self.data.dtype.name != "int32":
                    self.data = self.data.astype(dtype="int32", casting=casting)
            elif self.dataType == "float32":
                if self.data.dtype.name != "float32":
                    self.data = self.data.astype(dtype="float32", casting=casting)
            elif self.dataType == "complex64":
                if self.data.dtype.name != "complex64":
                    self.data = self.data.astype(dtype="complex64", casting=casting)
            elif self.dataType == "float64":
                if self.data.dtype.name != "float64":
                    self.data = self.data.astype(dtype="float64", casting=casting)
            else:
                wrong_data_type = True
        except TypeError:
            print(
                "Error most likely because trying to cast "
                + self.data.dtype.name
                + " array to "
                + self.dataType
                + " type. This may cause errors, so change argument dataType "
                "to an appropriate one."
            )
            raise
        if wrong_data_type:
            raise TypeError(
                "Data type " + self.dataType + " is not valid for EM"
                " format. Allowed types are: "
                + str(list(ImageIO.emDataTypeTab.values()))
            )

        # shape: use the argument, self.shape, or use default
        # if shape is not None: self.shape = shape
        # probably not needed (15.01.08)
        # if self.shape is None:
        #    if self.emHeader is not None:
        #        self.shape = self.emHeader[4:8]
        #    else:
        if self.shape is None:
            self.shape = copy(ImageIO.emDefaultShape)

        # use self.emHeader or the default em header
        if self.emHeader is None:
            self.emHeader = copy(ImageIO.emDefaultHeader)

        # add byteOrder, dataType and shape to the header
        try:
            self.emHeader[0] = ImageIO.emByteOrderTabInv[self.byteOrder]
            try:
                self.emHeader[3] = ImageIO.emDataTypeTabInv[self.dataType]
            except KeyError:
                print(
                    "Data type "
                    + self.dataType
                    + " is not valid for EM format."
                    + "Allowed types are: "
                    + str(list(ImageIO.emDataTypeTab.values()))
                )
                raise
            for k in range(len(self.shape)):
                self.emHeader[4 + k] = self.shape[k]
        except (AttributeError, LookupError):
            print("Need to specify byte order, data type and shape of the data.")
            raise

        # convert emHeader to a string and write it
        self.headerString = struct.pack(
            ImageIO.em["headerFormat"], *tuple(self.emHeader)
        )
        self.file_.write(self.headerString)

        # write data if exist
        if self.data is not None:
            self.writeData()
        self.file_.flush()

        return

    #####################################################
    #
    # MRC format
    #
    #####################################################

    # MRC file format properties
    mrc = {
        "headerSize": 1024,
        "headerFormat": "10i 6f 3i 3f 2i h 30s 4h 6f 6h 12f i 800s",
        "defaultByteOrder": machineByteOrder,
        #'defaultArrayOrder': 'FORTRAN',
        "defaultArrayOrder": "F",
        "defaultAxisOrder": (1, 2, 3),
    }
    mrcDefaultShape = [1, 1, 1]
    mrcDefaultPixel = [1, 1, 1]
    mrcDefaultHeader = (
        numpy.ones(3, "int32").tolist()
        + numpy.zeros(7, "int32").tolist()
        + numpy.zeros(6, "float32").tolist()
        + list(mrc["defaultAxisOrder"])
        + numpy.zeros(3, "float32").tolist()
        + numpy.zeros(2, "int32").tolist()
        + numpy.zeros(1, "int16").tolist()
        + [30 * b" "]
        + numpy.zeros(4, "int16").tolist()
        + numpy.zeros(6, "float32").tolist()
        + numpy.zeros(6, "int16").tolist()
        + numpy.zeros(12, "float32").tolist()
        + numpy.zeros(1, "int32").tolist()
        + [800 * b" "]
    )

    # type 3 not implemented, added imod type 6
    mrcDataTypeTab = {0: "ubyte", 1: "int16", 2: "float32", 4: "complex64", 6: "uint16"}
    mrcDataTypeTabInv = dict(
        list(zip(list(mrcDataTypeTab.values()), list(mrcDataTypeTab.keys())))
    )

    def readMRC(
        self,
        file=None,
        byteOrder=None,
        dataType=None,
        arrayOrder=None,
        shape=None,
        memmap=False,
    ):
        """
        Reads MRC file format.

        Parameters
        ----------
        file : str, optional
            Path to the file to read.
        byteOrder : {None, '<', '>'}, optional
            Byte order for reading the file. '<' means little endian and '>' means
            big endian.
        dataType : str, optional
            Data type to interpret the data.
        arrayOrder : str, optional
            Order in which to read the array.
        shape : tuple of int, optional
            Shape of the data array.
        memmap : bool, default False
            If true, use memory mapping to read data.
        """

        # open the file if needed
        self.checkFile(file=file, mode="rb")

        # parse arguments
        if byteOrder is not None:
            self.byteOrder = byteOrder
        if dataType is not None:
            self.dataType = dataType
        if arrayOrder is not None:
            self.arrayOrder = arrayOrder
        if shape is not None:
            self.shape = shape

        # use defaults if needed
        if self.byteOrder is None:
            self.byteOrder = ImageIO.mrc["defaultByteOrder"]
        if self.arrayOrder is None:
            self.arrayOrder = ImageIO.mrc["defaultArrayOrder"]

        # read the header
        self.readMRCHeader(file=self.file_)

        # read the data
        self.readData(shape=shape, memmap=memmap)

        return

    def readMRCHeader(self, file=None, byteOrder=None):
        """
        Reads the header of an MRC file.

        Parameters
        ----------
        file : str, optional
            Path to the file to read header from.
        byteOrder : {None, '<', '>'}, optional
            Byte order for reading the file. '<' means little endian and '>' means
            big endian.

        """

        # open the file if needed
        self.checkFile(file=file, mode="rb")

        # set byte order
        if byteOrder is not None:
            self.byteOrder = byteOrder
        if self.byteOrder is None:
            self.byteOrder = ImageIO.mrc["defaultByteOrder"]

        # read and unpack the header
        format = self.byteOrder + ImageIO.mrc["headerFormat"]
        self.headerString = self.file_.read(ImageIO.mrc["headerSize"])
        self.mrcHeader = list(struct.unpack(format, self.headerString))

        # check the data type
        data_type = ImageIO.mrcDataTypeTab.get(self.mrcHeader[3], None)
        if data_type is None:
            # byte order might be wrong: switch and unpack the header again
            if self.byteOrder == "<":
                header_byte_order = ">"
            else:
                header_byte_order = "<"
            format = header_byte_order + ImageIO.mrc["headerFormat"]
            self.mrcHeader = list(struct.unpack(format, self.headerString))

            # check new data type
            data_type = ImageIO.mrcDataTypeTab.get(self.mrcHeader[3], None)
            if data_type is None:
                raise ValueError(
                    "Could not determine the byte order or the data type "
                    + "of file "
                    + file
                )
            else:
                self.byteOrder = header_byte_order

        # parse header
        self.parseMRCHeader()

    def parseMRCHeader(self, header=None):
        """
        Parse the MRC header.

        If `header` is not provided, the function uses `self.mrcHeader`. It also
        sets `self.labels` if `self.headerString` is present. For this to work
        properly, `self.headerString` has to be consistent with the header
        (either argument or attribute).

        Parameters
        ----------
        header : array_like or None, optional
            The MRC header to parse. If None, `self.mrcHeader` is used.
        """
        if header is not None:
            self.mrcHeader = header

        # parse shape and data type
        self.shape = self.mrcHeader[0:3]  # C: z fastest changing
        self.dataType = ImageIO.mrcDataTypeTab[self.mrcHeader[3]]
        self.axisOrder = self.mrcHeader[16:19]  # read but not used

        # pixel size and length
        self.pixel = copy(self.mrcDefaultPixel)
        for ind in [0, 1, 2]:
            try:
                self.pixel[ind] = float(self.mrcHeader[ind + 10]) / (
                    10.0 * self.mrcHeader[ind]
                )
            except ZeroDivisionError:
                self.pixel[ind] = 1
        # self.pixel = [
        #    float(self.mrcHeader[10]) / self.mrcHeader[0],
        #    float(self.mrcHeader[11]) / self.mrcHeader[1],
        #    float(self.mrcHeader[12]) / self.mrcHeader[2]]
        self.length = [self.mrcHeader[10], self.mrcHeader[11], self.mrcHeader[12]]

        # labels (titles)
        try:
            self.n_labels = struct.unpack("i", self.headerString[220:224])[0]
            self.labels = []
            l_begin = 224
            for label_ind in range(self.n_labels):
                self.labels.append(self.headerString[l_begin : l_begin + 80])
                l_begin += 80
        except AttributeError:
            pass

        # read extended header if present
        self.extendedHeaderLength = self.mrcHeader[23]
        if header is None:
            self.extendedHeaderString = self.file_.read(self.extendedHeaderLength)

        return

    def writeMRC(
        self,
        file=None,
        header=None,
        byteOrder=None,
        shape=None,
        dataType=None,
        arrayOrder=None,
        length=None,
        pixel=None,
        data=None,
        extended=None,
        casting="unsafe",
    ):
        """
        Write data to an MRC file.

        This function writes both the header and data to an MRC file. If certain
        parameters are not provided, they are inferred from either other arguments,
        existing attributes, or default values.

        Parameters
        ----------
        file : str or file object, optional
            Path to the file or file object.
        header : array_like or None, optional
            MRC header to be written.
        byteOrder : str or None, optional
            Byte order of the MRC file.
        shape : tuple or None, optional
            Shape of the data.
        dataType : str or None, optional
            Type of data in the MRC file.
        arrayOrder : str or None, optional
            Order of the axes in the MRC file.
        length : list or None, optional
            Length in all dimensions in nm.
        pixel : float or None, optional
            Pixel size in nm.
        data : ndarray or None, optional
            Data to be written to the MRC file.
        extended : str or None, optional
            Extended header for the MRC file.
        casting : str, optional, default 'unsafe'
            Casting rule. Specifies how data should be casted if needed.
        """
        # open the file if needed
        self.checkFile(file=file, mode="wb")

        # set attributes from header
        if header is not None:
            self.parseMRCHeader(header=header)

        # buteOrder: use the argument, self.byteOrder, or the default value
        if byteOrder is not None:
            self.byteOrder = byteOrder
        if self.byteOrder is None:
            self.byteOrder = ImageIO.mrc["defaultByteOrder"]

        # arrayOrder: use the argument, self.arrayOrder, or the default value
        if arrayOrder is not None:
            self.arrayOrder = arrayOrder
        if self.arrayOrder is None:
            self.arrayOrder = ImageIO.mrc["defaultArrayOrder"]

        # pixel size: use the argument, self.pixel, or the default value
        if pixel is not None:
            self.pixel = pixel
        if self.pixel is None:
            self.pixel = copy(ImageIO.mrcDefaultPixel)

        # data: use the argument or the self.data
        # sets self.data, self.shape and self.dataType
        if data is not None:
            self.setData(data, shape=shape, pixel=self.pixel)
        # else:
        # adjust length for mrc header in case self.data was set before
        # self.fileFormat was set
        #    if (self.fileFormat is not None) and (self.fileFormat == 'mrc'):
        #        self.adjustLength(shape=None, pixel=self.pixel)

        # dataType: use the argument, self.dataType, or the mrcHeader value
        if dataType is not None:
            self.dataType = dataType
        if self.dataType is None:
            if self.mrcHeader is not None:
                self.dataType = ImageIO.mrcDataTypeTab[self.mrcHeader[3]]

        # unit8 and ubyte are the same
        if self.dataType == "uint8":
            self.dataType = "ubyte"

        # convert data to another dtype if needed
        wrong_data_type = False
        try:
            if (self.dataType == "ubyte") or (self.dataType == "uint8"):
                if self.data.dtype.name != "uint8":
                    self.data = self.data.astype(dtype="uint8", casting=casting)
            elif self.dataType == "int16":
                if self.data.dtype.name != "int16":
                    self.data = self.data.astype(dtype="int16", casting=casting)
            elif self.dataType == "float32":
                if self.data.dtype.name != "float32":
                    self.data = self.data.astype(dtype="float32", casting=casting)
            elif self.dataType == "complex64":
                if self.data.dtype.name != "complex64":
                    self.data = self.data.astype(dtype="complex64", casting=casting)
            else:
                wrong_data_type = True
        except TypeError:
            print(
                "Error most likely because trying to cast "
                + self.data.dtype.name
                + " array to "
                + str(self.dataType)
                + " type. This may cause errors, so change argument dataType "
                "to an appropriate one."
            )
            raise
        if wrong_data_type:
            raise TypeError(
                "Data type " + str(self.dataType) + " is not valid for MRC"
                " format. Allowed types are: "
                + str(list(ImageIO.mrcDataTypeTab.values()))
            )

        # axisOrder: self.axisOrder or default
        if self.axisOrder is None:
            self.axisOrder = ImageIO.mrc["defaultAxisOrder"]

        # shape: use the argument, self.shape, get from header, or use default
        # if shape is not None: self.shape = shape
        if self.shape is None:
            self.shape = copy(ImageIO.mrcDefaultShape)

        # self.shape has to have length 3
        if len(self.shape) < 3:
            if isinstance(self.shape, list):
                self.shape = self.shape + [1] * (3 - len(self.shape))
            elif isinstance(self.shape, tuple):
                self.shape = self.shape + (1,) * (3 - len(self.shape))

        # make self.pixel a list
        try:
            if not isinstance(self.pixel, (list, tuple)):
                self.pixel = [pixel] * len(self.shape)
        except (AttributeError, LookupError):
            print("Need to specify shape of the data.")
            raise

        # length: use the argument, self.length, or shape * pixel_in_A
        if length is not None:
            self.length = length
        if self.length is None:
            try:
                self.adjustLength()
                # self.length = 10 * numpy.asarray(self.shape) \
                #    * numpy.asarray(self.pixel)
            except (AttributeError, LookupError):
                print("Need to specify shape of the data.")
                raise

        # use header, self.mrcHeader or the default mrc header
        if header is not None:
            self.mrcHeader = header
        if self.mrcHeader is None:
            self.mrcHeader = copy(ImageIO.mrcDefaultHeader)
        if extended is not None:
            self.extended = extended

        # add shape, data type and axisOrder to the header
        try:
            for k in range(len(self.shape)):
                self.mrcHeader[k] = self.shape[k]
                self.mrcHeader[k + 7] = self.shape[k]
                self.mrcHeader[k + 10] = self.length[k]
            try:
                self.mrcHeader[3] = ImageIO.mrcDataTypeTabInv[self.dataType]
            except KeyError:
                print(
                    "Data type "
                    + str(self.dataType)
                    + " is not valid for "
                    + "MRC format. Allowed types are: "
                    + str(list(ImageIO.mrcDataTypeTab.values()))
                )
                raise
            self.mrcHeader[16:19] = self.axisOrder
        except (AttributeError, LookupError):
            print("Need to specify data type and shape of the data.")
            raise

        # add min max and mean values
        if self.data is not None:
            self.mrcHeader[19] = self.data.min()
            self.mrcHeader[20] = self.data.max()
            self.mrcHeader[21] = self.data.mean()

        # convert header to a string and write it
        self.headerString = struct.pack(
            ImageIO.mrc["headerFormat"], *tuple(self.mrcHeader)
        )
        if extended is not None:
            self.headerString = self.headerString + extended
        self.file_.write(self.headerString)

        # write data if exist
        if self.data is not None:
            self.writeData()
        self.file_.flush()

        return

    def adjustLength(self, shape=None, pixel=None):
        """
        Calculate the length based on shape and pixel size.

        This function computes the length of the data in Angstroms, which is used
        in the MRC header. It's specifically designed for MRC files.

        Parameters
        ----------
        shape : tuple or None, optional
            Shape of the data.
        pixel : float or None, optional
            Pixel size in nm.
        """
        # set variables
        if shape is None:
            shape = self.shape
        if pixel is None:
            pixel = self.pixel

        # calculate length in all 3 dimensions
        shape = numpy.asarray(shape)
        if len(shape) < 3:
            shape = numpy.concatenate((shape, (3 - len(shape)) * [1]))
        self.length = 10 * shape * numpy.asarray(pixel)

    #####################################################
    #
    # Tiff file format
    #
    ######################################################

    # def readTiff()

    #####################################################
    #
    # Raw file format
    #
    ######################################################

    # raw file format properties
    raw = {
        "defaultHeaderSize": 0,
        "defaultByteOrder": machineByteOrder,
        #'defaultArrayOrder': 'FORTRAN'
        "defaultArrayOrder": "F",
    }

    def readRaw(
        self,
        file=None,
        dataType=None,
        shape=None,
        byteOrder=None,
        arrayOrder=None,
        headerSize=None,
        memmap=False,
    ):
        """
        Read data from a raw file.

        Parameters
        ----------
        file : str or file object, optional
            Path to the file or file object.
        dataType : str or None, optional
            Type of data in the raw file.
        shape : tuple or None, optional
            Shape of the data.
        byteOrder : str or None, optional
            Byte order of the raw file.
        arrayOrder : str or None, optional
            Order of the axes in the raw file.
        headerSize : int or None, optional
            Size of the header in the raw file.
        memmap : bool, optional, default False
            Whether to memory-map the file.

        """

        # open the file if needed
        self.checkFile(file=file, mode="rb")

        # set defaults
        self.byteOrder = ImageIO.raw["defaultByteOrder"]
        self.arrayOrder = ImageIO.raw["defaultArrayOrder"]

        # parse arguments
        if file is not None:
            self.file_ = file
        if byteOrder is not None:
            self.byteOrder = byteOrder
        if dataType is not None:
            self.dataType = dataType
        if arrayOrder is not None:
            self.arrayOrder = arrayOrder
        if shape is not None:
            self.shape = shape
        if headerSize is not None:
            self.rawHeaderSize = headerSize

        # read header
        self.readRawHeader(file=self.file_, size=self.rawHeaderSize)

        # read data
        self.readData(shape=shape, memmap=memmap)

        return

    def readRawHeader(self, file=None, size=None):
        """
        Read the header from a raw file.

        Parameters
        ----------
        file : str or file object, optional
            Path to the file or file object.
        size : int or None, optional
            Size of the header in the raw file.
        """
        # open the file if needed
        self.checkFile(file=file, mode="rb")

        # determine header size
        if size is not None:
            self.rawHeaderSize = size
        elif self.rawHeaderSize is None:
            self.rawHeaderSize = self.raw["defaultHeaderSize"]

        # read the header
        if (size is not None) and (size > 0):
            self.headerString = self.file_.read(self.rawHeaderSize)
        else:
            self.headerString = ""

    def writeRaw(
        self,
        file=None,
        header=None,
        data=None,
        shape=None,
        dataType=None,
        byteOrder=None,
        arrayOrder=None,
        casting="unsafe",
    ):
        """
        Writes raw data.

        Parameters
        ----------
        file : str or file-like, optional
            File path or file handle.
        header : str or bytes, optional
            File header.
        data : array-like, optional
            Data to be written.
        shape : tuple, optional
            Shape of the data.
        dataType : dtype, optional
            Data type.
        byteOrder : str, optional
            Byte order.
        arrayOrder : str, optional
            Array order.
        casting : str, optional
            Casting method. Default is "unsafe".

        Raises
        ------
        TypeError
            If data type is incompatible.
        """

        # open the file if needed
        self.checkFile(file=file, mode="wb")

        # set defaults
        self.arrayOrder = ImageIO.raw["defaultArrayOrder"]
        self.byteOrder = ImageIO.raw["defaultByteOrder"]

        # parse arguments
        if file is not None:
            self.file_ = file
        if data is not None:
            self.setData(data, shape=shape)  # sets self.shape also
        if byteOrder is not None:
            self.byteOrder = byteOrder
        if arrayOrder is not None:
            self.arrayOrder = arrayOrder

        # data type
        if dataType is not None:
            self.dataType = dataType
        else:
            self.dataType = self.data.dtype.name
        if self.dataType != self.data.dtype.name:
            try:
                self.data = self.data.astype(dtype=self.dataType, casting=casting)
            except TypeError:
                print(
                    "Error most likely because trying to cast "
                    + self.data.dtype.name
                    + " array to "
                    + self.dataType
                    + " type. This may cause errors, so change argument "
                    "dataType to an appropriate one."
                )
                raise

        # write header
        if header is not None:
            self.file_.write(header)

        # write data
        self.writeData()

        return

    ########################################################
    #
    # Common read/write methods
    #
    ########################################################

    def setData(self, data, shape=None, pixel=None):
        """
        Reshapes and saves data as an attribute.

        Parameters
        ----------
        data : array-like
            Image data.
        shape : tuple, optional
            Image shape.
        pixel : float, optional
            Pixel size in nm.
        """

        # make shape of length 3
        if (shape is not None) and (len(shape) < 3):
            if len(shape) == 2:
                shape = (shape[0], shape[1], 1)
            elif len(shape) == 1:
                shape = (shape[0], 1, 1)

        # set data and shape
        self.data = data
        if self.data is not None:
            if shape is not None:
                self.data = self.data.reshape(shape)
            self.shape = self.data.shape
            self.dataType = self.data.dtype.name

        # adjust length for mrc files
        if (self.fileFormat is not None) and (self.fileFormat == "mrc"):
            self.adjustLength(shape=None, pixel=pixel)

    def readData(self, shape=None, memmap=False):
        """
        Reads data from an image file.

        Parameters
        ----------
        shape : tuple, optional
            Shape of the image.
        memmap : bool, optional
            If True, creates a memory map. Default is False.

        Raises
        ------
        ValueError
            If byte order is incompatible with memory map.

        """

        # check if there's an extended header
        try:
            ext_head_len = self.extendedHeaderLength
        except AttributeError:
            ext_head_len = 0
        total_head_len = len(self.headerString) + ext_head_len

        # read data in numpy.ndarray
        if memmap:
            self.data = numpy.memmap(
                self.file_,
                mode="r",
                shape=tuple(self.shape),
                dtype=self.dataType,
                offset=total_head_len,
                order=self.arrayOrder,
            )
            self.memmap = True
        else:
            self.data = numpy.fromfile(file=self.file_, dtype=self.dataType)
            self.memmap = False

        # reshape data
        if self.arrayOrder is None:
            self.arrayOrder = self.defaultArrayOrder
        if shape is not None:
            self.shape = shape
        self.data = self.data.reshape(self.shape, order=self.arrayOrder)

        # chage byte order (to little-endian) if needed
        if self.byteOrder == ">":
            if memmap:
                raise ValueError(
                    "Can not change byte order to '>' because this file is "
                    + " read using memory map. Run "
                    + "without memory map (set memmap argument to False)."
                )
            else:
                self.data = self.data.byteswap(True)

        return

    def writeData(self):
        """
        Writes data in numpy.ndarray format to an image file.

        Raises
        ------
        AttributeError, LookupError
            If required attributes are not set.

        """

        # change dataType, byteOrder and arrayOrder if needed
        try:
            if self.data.dtype != self.dataType:
                self.data = numpy.asarray(self.data, dtype=self.dataType)
            if self.byteOrder == ">":
                self.data = self.data.byteswap(True)
            self.data = self.data.reshape(self.data.size, order=self.arrayOrder)
        except (AttributeError, LookupError):
            print("Need to specify data.")
            raise

        # write
        self.data.tofile(file=self.file_)

        # reshape data back to original shape
        self.data = self.data.reshape(self.shape, order=self.arrayOrder)

    def setFileFormat(self, fileFormat=None, file_=None):
        """
        Sets the file format.

        Parameters
        ----------
        fileFormat : str, optional
            File format.
        file_ : str, optional
            File name.
        """
        if fileFormat is not None:
            # fileFormat argiment given
            self.fileFormat = fileFormat

        else:
            # parse file_ argument
            if file_ is None:
                file_ = self.fileName

            # find the extension of file_ to determine the format
            if isinstance(file_, basestring):  # file argument is a file name
                splitFileName = os.path.splitext(file_)
                extension = splitFileName[-1].lstrip(".")
                self.fileFormat = ImageIO.fileFormats.get(extension)
            else:
                # fileFormat not set here, raise an exception later if needed
                pass

        return

    def checkFile(self, file, mode):
        """
        Checks and possibly opens the file.

        Parameters
        ----------
        file : str or file-like
            File path or file handle.
        mode : str
            File mode, as in the open() function.

        Raises
        ------
        IOError
            If file is neither a string nor a file object.
        """

        # use self.fileName if file_ is None
        if file is None:
            # print("file is None")
            file = self.fileName

        # needed because file type not defined in Python3
        try:
            filetypes = (io.IOBase, file)
        except NameError:
            filetypes = io.IOBase

        # open the file if not opened already
        if isinstance(file, basestring):  # file_ is a string
            self.fileName = file
            self.file_ = open(file, mode)

        elif isinstance(file, filetypes):  # file already open
            self.file_ = file

        else:
            raise IOError(
                "Argument file_: "
                + str(file)
                + "is neither a string nor a file object"
            )

        return

    ########################################################
    #
    # Header manipulations
    #
    ########################################################

    def getTiltAngle(self):
        """
        Returns tilt angle in degrees.

        Raises
        ------
        ValueError
            If tilt angle cannot be retrieved for the file format.

        Returns
        -------
        float
            Tilt angle.
        """

        # ToDo: get from header directly?
        if self.fileFormat == "em":
            return self._tiltAngle / 1000.0

        else:
            raise ValueError(
                "Sorry can't get tilt angle for " + self.fileFormat + " file."
            )

    def setTiltAngle(self, angle):
        """
        Set tilt angle for EM format files.

        Sets the internal `_tiltAngle` attribute and updates the corresponding value
        in `emHeader`.

        Parameters
        ----------
        angle : float
            Tilt angle to set (in degrees).

        Raises
        ------
        ValueError
            If the file format is not 'em'.
        """
        if self.fileFormat == "em":
            # set the attribute
            self._tiltAngle = angle * 1000

            # put in the emHeader
            self.putInEMHeader(name="_tiltAngle", value=self._tiltAngle)

        else:
            raise ValueError(
                "Sorry, can't get tilt angle for " + self.fileFormat + " file."
            )

    tiltAngle = property(
        fget=getTiltAngle, fset=setTiltAngle, doc="Tilt angle (in deg)"
    )

    def getPixelsize(self, diff=1e-6):
        """
        Return pixel size at the specimen level in nanometers.

        For mrc files, it returns a single pixelsize if it's the same for all
        dimensions, otherwise a list of pixelsizes for each dimension is returned.
        Pixelsizes are considered the same if they do not differ more than the
        specified difference.

        Parameters
        ----------
        diff : float, optional
            The threshold for pixel size difference to be considered the same, by
            default 1e-6.

        Returns
        -------
        float or list of float
            Pixel size(s) in nm.

        Raises
        ------
        ValueError
            If the file format is neither 'em' nor 'mrc'.
        """
        if self.fileFormat == "em":
            return self._pixelsize / 1000.0
        elif self.fileFormat == "mrc":
            if isinstance(self.pixel, (int, float)):
                return self.pixel
            else:
                same = [numpy.abs(self.pixel[0] - pix) < diff for pix in self.pixel]
                if numpy.asarray(same).all():
                    return self.pixel[0]
                else:
                    return self.pixel
        else:
            raise ValueError(
                "Sorry can't get pixel size for " + self.fileFormat + " file."
            )

    pixelsize = property(fget=getPixelsize, doc="Pixel size (at specimen level) in nm")

    def fix(self, mode=None, microscope=None):
        """
        Fix wrong values in both header and data.

        Parameters
        ----------
        mode : str, optional
            Determines which values are fixed. Possible values include
            'polara_fei-tomo', 'krios_fei-tomo', 'cm300', by default None.
        microscope : str, optional
            Specified when mode is 'polara_fei-tomo'. Currently accepted values are:
            'polara-1_01-07', 'polara-1_01-09' and 'polara-2_01-09', by default None.

        See Also
        --------
        fixHeader : Fix values only in the header.
        """

        self.fixHeader(mode=mode, microscope=microscope)

        # fix data to be implemented

    def fixHeader(self, mode=None, microscope=None):
        """
        Fix wrong values in the microscope image header.

        Parameters
        ----------
        mode : str, optional
            Determines which values are fixed. Possible values include
            'polara_fei-tomo', 'krios_fei-tomo', 'cm300', by default None.
        microscope : str, optional
            Specified when mode is 'polara_fei-tomo'. Currently accepted values are:
            'polara-1_01-07', 'polara-1_01-09' and 'polara-2_01-09', by default None.

        Raises
        ------
        ValueError
            If the given mode is not recognized for the current file format.
        """

        if self.fileFormat == "em":
            if mode == "polara_fei-tomo":
                # put voltage
                self.putInEMHeader(name="voltage", value=300000)

                # put cs
                self.putInEMHeader(name="cs", value=2000)

                # put CCD pixel size
                ccd_pixelsize = microscope_db.ccd_pixelsize[microscope]
                self.putInEMHeader(name="ccdPixelsize", value=ccd_pixelsize)

                # put CCD length (pixel size * number of pixels)
                self.putInEMHeader(
                    name="ccdLength",
                    value=microscope_db.n_pixels[microscope] * ccd_pixelsize,
                )

                # get nominal magnification and determine (real) pixel size
                mag = self.getFromEMHeader("magnification")
                pixelsize = microscope_db.pixelsize[microscope][mag]
                self.putInEMHeader(name="_pixelsize", value=pixelsize)

            elif mode == "krios_fei-tomo":
                # put voltage
                self.putInEMHeader(name="voltage", value=300000)

                # put cs
                self.putInEMHeader(name="cs", value=2000)

                # put CCD pixel size
                # ccd_pixelsize = microscope_db.ccd_pixelsize[microscope]
                # self.putInEMHeader(name='ccdPixelsize', value=ccd_pixelsize)

                # put CCD length (pixel size * number of pixels)
                # The value might be correct
                # self.putInEMHeader(
                #    name='ccdLength',
                #    value=microscope_db.n_pixels[microscope] * ccd_pixelsize)

                # get nominal magnification and determine (real) pixel size
                # not needed because correct value there already
                # mag = self.getFromEMHeader('magnification')
                # pixelsize = microscope_db.pixelsize[microscope][mag]
                # self.putInEMHeader(name='_pixelsize', value=pixelsize)

            elif mode == "cm300":
                microscope = "cm300"

                # put voltage
                self.putInEMHeader(name="voltage", value=300000)

                # put cs
                self.putInEMHeader(name="cs", value=2000)

                # get magnification and determine (real) pixel size
                mag = self.getFromEMHeader("magnification")
                nom_mag = microscope_db.nominal_mag[microscope][mag]
                pixelsize = microscope_db.pixelsize[microscope][nom_mag]
                self.putInEMHeader(name="_pixelsize", value=pixelsize)

                # correct em code
                self.putInEMHeader(name="emCode", value=0)

                # put CCD pixel size
                ccd_pixelsize = microscope_db.ccd_pixelsize[microscope]
                self.putInEMHeader(name="ccdPixelsize", value=ccd_pixelsize)

                # put CCD length (pixel size * number of pixels)
                self.putInEMHeader(
                    name="ccdLength",
                    value=microscope_db.n_pixels[microscope] * ccd_pixelsize,
                )

            elif mode is None:
                pass

            else:
                raise ValueError(
                    "Sorry, mode: "
                    + str(mode)
                    + " is not recognized for an "
                    + self.fileFormat
                    + " file."
                )

        elif self.fileFormat == "mrc":
            if mode is None:
                pass

            else:
                raise ValueError(
                    "Sorry, mode: "
                    + mode
                    + " is not recognized for an "
                    + self.fileFormat
                    + " file."
                )

        elif self.fileFormat == "raw":
            if mode is None:
                pass

            else:
                raise ValueError(
                    "Sorry, mode: "
                    + mode
                    + " is not recognized for an "
                    + self.fileFormat
                    + " file."
                )

        else:
            raise ValueError(
                "Sorry, file format: " + self.fileFormat + " is not recognized."
            )

    def getFromEMHeader(self, name):
        """
        Get the value of a variable from the EM header.

        Parameters
        ----------
        name : str
            The name of the variable to retrieve from the header.

        Returns
        -------
        variable_type
            The value of the specified variable from the header.

        Notes
        -----
        Alternatively, self.name can be used, but the best approach is yet to
        be determined.
        """

        # find position of name in self.emHeaderFields
        ind = 0
        reg = re.compile(name + "\\b")
        for field in self.emHeaderFields:
            if reg.match(field) is not None:
                break
            ind += 1

        # return the value
        return self.emHeader[ind]

    def putInEMHeader(self, name, value):
        """
        Update a value in the EM header.

        Parameters
        ----------
        name : str
            The name of the variable to update in the header.
        value : variable_type
            The value to set for the specified variable in the header.

        Notes
        -----
        This function updates the specified value in the header in-place.
        """

        # find position of name in self.emHeaderFields
        ind = 0
        reg = re.compile(name + "\\b")
        for field in self.emHeaderFields:
            if reg.match(field) is not None:
                break
            ind += 1

        # put the value in
        self.emHeader[ind] = value
