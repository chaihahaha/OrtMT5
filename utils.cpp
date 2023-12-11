#include "utils.h"

std::wstring str2wstr(std::string s)
{
    icu::UnicodeString str = icu::UnicodeString::fromUTF8(icu::StringPiece(s.c_str()));
    //std::cout << "unicode" << str << std::endl;
    
    int32_t requiredSize;
    UErrorCode error = U_ZERO_ERROR;
    
    wchar_t buffer[8192];
    u_strToWCS(buffer, 8192, NULL, str.getBuffer(), str.length(), &error);
    std::wstring wstr(buffer);
    //std::wcout << L"converted" << wstr << " " << requiredSize << std::endl;
    return wstr;
}