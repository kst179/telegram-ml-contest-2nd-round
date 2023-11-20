#ifndef TGLANG_H
#define TGLANG_H

/**
 * Library for determining programming or markup language of a code snippet.
 */
 
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#  ifdef tglang_EXPORTS
#    define TGLANG_EXPORT __declspec(dllexport)
#  else
#    define TGLANG_EXPORT __declspec(dllimport)
#  endif
#else
#  define TGLANG_EXPORT __attribute__((visibility("default")))
#endif

/**
 * List of supported languages.
 */
enum TglangLanguage {
    TGLANG_LANGUAGE_OTHER,
    TGLANG_LANGUAGE_C,
    TGLANG_LANGUAGE_CPLUSPLUS,
    TGLANG_LANGUAGE_CSHARP,
    TGLANG_LANGUAGE_CSS,
    TGLANG_LANGUAGE_DART,
    TGLANG_LANGUAGE_DOCKER,
    TGLANG_LANGUAGE_FUNC,
    TGLANG_LANGUAGE_GO,
    TGLANG_LANGUAGE_HTML,
    TGLANG_LANGUAGE_JAVA,
    TGLANG_LANGUAGE_JAVASCRIPT,
    TGLANG_LANGUAGE_JSON,
    TGLANG_LANGUAGE_KOTLIN,
    TGLANG_LANGUAGE_LUA,
    TGLANG_LANGUAGE_NGINX,
    TGLANG_LANGUAGE_OBJECTIVE_C,
    TGLANG_LANGUAGE_PHP,
    TGLANG_LANGUAGE_POWERSHELL,
    TGLANG_LANGUAGE_PYTHON,
    TGLANG_LANGUAGE_RUBY,
    TGLANG_LANGUAGE_RUST,
    TGLANG_LANGUAGE_SHELL,
    TGLANG_LANGUAGE_SOLIDITY,
    TGLANG_LANGUAGE_SQL,
    TGLANG_LANGUAGE_SWIFT,
    TGLANG_LANGUAGE_TL,
    TGLANG_LANGUAGE_TYPESCRIPT,
    TGLANG_LANGUAGE_XML,
};

/**
 * Detects programming or markup language of a code snippet.
 * \param[in] text Text of a code snippet. A null-terminated string in UTF-8 encoding.
 * \return detected programming language.
 */
TGLANG_EXPORT enum TglangLanguage tglang_detect_programming_language(const char *text);

#ifdef __cplusplus
}
#endif

#endif
